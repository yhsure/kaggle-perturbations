import torch 
import copy 
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch.nn.functional as F

def train_dgd(train_loader, val_loader, test_loader, model, gmm, rep, val_rep, test_rep, lrs, betas=(0.9, 0.999), 
wd=0, n_epochs=150, plot_step=1000, device="cpu", print_loss_step=10, plot=False, patience=30, start_saving=0, task_weight=1, mean_reg=False):
    # note: start_saving is the epoch at which the model starts saving and loop cant return before that
    model.to(device)
    rep.to(device)
    val_rep.to(device)
    test_rep.to(device)
    if gmm is not None:
        gmm.to(device)

    nsample = len(train_loader.dataset)
    nsample_val = len(val_loader.dataset)
    nsample_test = len(test_loader.dataset)
    out_dim = len(train_loader.dataset.genes)
    latent = model.fc_layers[0].in_features - model.n_conditional_vars

    model_optimizer = torch.optim.Adam(model.parameters(), lr=lrs[0], weight_decay=wd, betas=betas)
    if gmm is not None:
        gmm_optimizer = torch.optim.Adam(gmm.parameters(), lr=lrs[2], weight_decay=wd,betas=betas)
    rep_optimizer = torch.optim.Adam(rep.parameters(), lr=lrs[1], weight_decay=wd,betas=betas)
    valrep_optimizer = torch.optim.Adam(val_rep.parameters(), lr=lrs[1], weight_decay=wd,betas=betas)
    testrep_optimizer = torch.optim.Adam(test_rep.parameters(), lr=lrs[1], weight_decay=wd,betas=betas)

    # model_optimizer = torch.optim.RMSprop(model.parameters(), lr=lrs[0], weight_decay=wd)
    # if gmm is not None:
    #     gmm_optimizer = torch.optim.RMSprop(gmm.parameters(), lr=lrs[2], weight_decay=wd)
    # rep_optimizer = torch.optim.RMSprop(rep.parameters(), lr=lrs[1], weight_decay=wd)
    # valrep_optimizer = torch.optim.RMSprop(val_rep.parameters(), lr=lrs[1], weight_decay=wd)
    # testrep_optimizer = torch.optim.RMSprop(test_rep.parameters(), lr=lrs[1], weight_decay=wd)


    train_avg = []
    recon_avg = []
    task_avg = []
    val_avg = []
    recon_val_avg = []
    task_val_avg = []
    dist_avg = []
    dist_val_avg = []

    best_model = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    best_epoch = -1
    patience_counter = 0

    for epoch in range(n_epochs):

        train_avg.append(0)
        val_avg.append(0)
        recon_avg.append(0)
        recon_val_avg.append(0)
        task_avg.append(0)
        task_val_avg.append(0)
        if gmm is not None:
            dist_avg.append(0)
            dist_val_avg.append(0)

        # train
        rep_optimizer.zero_grad()
        for i, data, data_max, target_class, conditional_vars in train_loader:
            i, data, data_max, target_class, conditional_vars = i.to(device), data.to(device), data_max.to(device), target_class.to(device), conditional_vars.to(device)

            if gmm is not None:
              gmm_optimizer.zero_grad()
            model_optimizer.zero_grad()

            # forward pass and unpack
            z = rep(i)
            y = model(z, conditional_vars if model.n_conditional_vars > 0 else None)
            label_preds = y[:, out_dim:]
            y = y[:, :out_dim]

            # loss
            recon_loss_x = model.nb.loss(data, data_max, y).sum()

            #ce_loss = F.cross_entropy(label_preds, target_class, reduction='sum')    

            if gmm is not None:
                gmm_error = - gmm(z).sum()
                loss = recon_loss_x + gmm_error #+ task_weight * ce_loss
            else:
                loss = recon_loss_x #+ task_weight * ce_loss

            # backward pass and step
            loss.backward()
            if gmm is not None:
              gmm_optimizer.step()
            model_optimizer.step()

            train_avg[-1] += loss.item()
            recon_avg[-1] += recon_loss_x.item()
            # task_avg[-1] += ce_loss.item()
            if gmm is not None:
              dist_avg[-1] += gmm_error.item()
        rep_optimizer.step()

        # val
        valrep_optimizer.zero_grad()
        for i, data, data_max, target_class, conditional_vars in val_loader:
            i, data, data_max, target_class, conditional_vars = i.to(device), data.to(device), data_max.to(device), target_class.to(device), conditional_vars.to(device)
            z = val_rep(i)
            y = model(z, conditional_vars if model.n_conditional_vars > 0 else None)
            label_preds = y[:, out_dim:]
            y = y[:, :out_dim]

            recon_loss_x = model.nb.loss(data, data_max, y).sum()
            # ce_loss = F.cross_entropy(label_preds, target_class, reduction='sum')

            if gmm is not None:
                gmm_error = - gmm(z).sum()
                loss = recon_loss_x + gmm_error #+ task_weight * ce_loss
            else:
                loss = recon_loss_x #+ task_weight * ce_loss
            loss.backward()

            val_avg[-1] += loss.item()
            recon_val_avg[-1] += recon_loss_x.item()
            # task_val_avg[-1] += ce_loss.item()
            if gmm is not None:
                dist_val_avg[-1] += gmm_error.item()
        valrep_optimizer.step()

        # test
        testrep_optimizer.zero_grad()
        for i, data, data_max, target_class, conditional_vars in test_loader:
            i, data, data_max, target_class, conditional_vars = i.to(device), data.to(device), data_max.to(device), target_class.to(device), conditional_vars.to(device)
            z = test_rep(i)
            y = model(z, conditional_vars if model.n_conditional_vars > 0 else None)
            label_preds = y[:, out_dim:]
            y = y[:, :out_dim]

            recon_loss_x = model.nb.loss(data, data_max, y).sum()
            # ce_loss = F.cross_entropy(label_preds, target_class, reduction='sum')

            if gmm is not None:
                gmm_error = - gmm(z).sum()
                loss = recon_loss_x + gmm_error #+ task_weight * ce_loss
            else:
                loss = recon_loss_x #+ task_weight * ce_loss
            loss.backward()
        testrep_optimizer.step()


        train_avg[-1] /= (nsample*out_dim)
        val_avg[-1] /= (nsample_val*out_dim)
        recon_avg[-1] /= (nsample*out_dim)
        recon_val_avg[-1] /= (nsample_val*out_dim)
        task_avg[-1] /= (nsample*out_dim)
        task_val_avg[-1] /= (nsample_val*out_dim)
        if gmm is not None:
            dist_avg[-1] /= (nsample*latent*gmm.Nmix)
            dist_val_avg[-1] /= (nsample_val*latent*gmm.Nmix)


        # print epoch stats
        if epoch % print_loss_step == 0:
            print(('Epoch {:>3}  T loss: {:.4f}  V loss: {:.4f}' + \
                    '  T Recon: {:.4f}  V Recon: {:.4f}' + \
                    '  T Task: {:.7f}  V Task: {:.7f}').format(
                epoch, train_avg[-1], val_avg[-1], recon_avg[-1], 
                recon_val_avg[-1], task_avg[-1], 0))#task_val_avg[-1]))
            
        # Early stopping
        if val_avg[-1] < best_val_loss:
            best_val_loss = val_avg[-1]
            if epoch >= start_saving:
                best_model = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience and epoch >= start_saving:
            print("Early stopping")
            break


        if epoch % plot_step == 0 and epoch != 0 and epoch != n_epochs-1 and plot:
            # plot_negbino(model, tdata=train_loader.dataset.data, r=rep, e=epoch, gmm=gmm, 
            #              scaling='library',library=train_loader.dataset.data_max, labels=train_loader.dataset.obs.cell_type.cat.codes)
            #plot_gene_dists(tdata=train_loader.dataset.data, y=y, lib=scaling_factors, ngenes=4)
            history = pd.DataFrame({'epoch': np.arange(len(train_avg)),
                                    'loss': train_avg,
                                    '-log p(x|z)': recon_avg,
                                    'log p(z)': dist_avg,
                                    # 'task': task_avg,
                                    'type': 'train'})
            temp = pd.DataFrame({'epoch': np.arange(len(train_avg)),
                                    'loss': val_avg,
                                    '-log p(x|z)': recon_val_avg,
                                    'log p(z)': dist_val_avg,
                                    # 'task': task_val_avg,
                                    'type': 'test'}, index=[x+len(train_avg) for x in np.arange(len(train_avg))])
            history = pd.concat([history, temp])

            scatterplot_sizes = (6, 6, 40)
            fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(20, 5))
            sns.lineplot(data=history, x='epoch', y='loss', hue='type', ax=ax[0])
            sns.lineplot(data=history, x='epoch', y='-log p(x|z)', hue='type', ax=ax[1])
            sns.lineplot(data=history, x='epoch', y='log p(z)', hue='type', ax=ax[2])

            pca = PCA(n_components=2)
            pca_data = pd.DataFrame(data=rep.z.cpu().detach().numpy())
            pca_data['type'] = 'representation'
            pca_data['size'] = scatterplot_sizes[0]
            if gmm is not None:
                sampled = gmm.sample(500)
                temp = pd.DataFrame(data=sampled.cpu().detach().numpy())
                temp['type'] = 'gmm'
                temp['size'] = scatterplot_sizes[1]
                pca_data = pd.concat([pca_data, temp])
                if len(gmm.mean.cpu().detach().numpy().shape) > 1:
                    temp = pd.DataFrame(data=gmm.mean.cpu().detach().numpy())
                    temp['type'] = 'gmm_mean'
                    temp['size'] = scatterplot_sizes[2]
                    pca_data = pd.concat([pca_data, temp])
                pca.fit(pca_data.iloc[:, :-2])
                projected = pca.fit_transform(pca_data.iloc[:, :-2])
                projected_norm = (projected - np.mean(projected.flatten())) / np.std(projected.flatten())
                pca_data2 = pd.DataFrame(data=projected_norm, columns=['PC1', 'PC2'])
                pca_data2['type'] = pca_data['type'].values
                pca_data2['size'] = pca_data['size'].values
                sns.scatterplot(x='PC1', y='PC2', data=pca_data2, hue='type', size='size', ax=ax[3], sizes=(scatterplot_sizes[0], scatterplot_sizes[2]))
                ax[3].legend(bbox_to_anchor=(1.05, 1.), loc='upper left')
            else:
                pca.fit(pca_data.iloc[:, :-2])
                projected = pca.fit_transform(pca_data.iloc[:, :-2])
                projected_norm = (projected - np.mean(projected.flatten())) / np.std(projected.flatten())
                pca_data2 = pd.DataFrame(data=projected_norm, columns=['PC1', 'PC2'])
                pca_data2['type'] = pca_data['type'].values
                pca_data2['size'] = pca_data['size'].values
                sns.scatterplot(x='PC1', y='PC2', data=pca_data2, hue='type', size='size', ax=ax[3], sizes=(scatterplot_sizes[0], scatterplot_sizes[2]))
                ax[3].legend(bbox_to_anchor=(1.05, 1.), loc='upper left')
            plt.tight_layout()
            plt.show()


    if gmm is not None:
        history = pd.DataFrame({'epoch': np.arange(len(train_avg)),
                                'loss': train_avg,
                                '-log p(x|z)': recon_avg,
                                'log p(z)': dist_avg,
                                # 'task': task_avg,
                                'type': 'train'})
        temp = pd.DataFrame({'epoch': np.arange(len(train_avg)),
                                'loss': val_avg,
                                '-log p(x|z)': recon_val_avg,
                                'log p(z)': dist_val_avg,
                                # 'task': task_val_avg,
                                'type': 'test'}, index=[x+len(train_avg) for x in np.arange(len(train_avg))])
        history = pd.concat([history, temp])

        scatterplot_sizes = (6, 6, 40)
        fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(20, 5))
        sns.lineplot(data=history, x='epoch', y='loss', hue='type', ax=ax[0])
        sns.lineplot(data=history, x='epoch', y='-log p(x|z)', hue='type', ax=ax[1])
        sns.lineplot(data=history, x='epoch', y='log p(z)', hue='type', ax=ax[2])
    else:
        history = pd.DataFrame({'epoch': np.arange(len(train_avg)),
                                'loss': train_avg,
                                '-log p(x|z)': recon_avg,
                                # 'task': task_avg,
                                'type': 'train'})
        temp = pd.DataFrame({'epoch': np.arange(len(train_avg)),
                                'loss': val_avg,
                                '-log p(x|z)': recon_val_avg,
                                # 'task': task_val_avg,
                                'type': 'test'}, index=[x+len(train_avg) for x in np.arange(len(train_avg))])
        history = pd.concat([history, temp])

        scatterplot_sizes = (6, 6, 40)
        fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(20, 5))
        sns.lineplot(data=history, x='epoch', y='loss', hue='type', ax=ax[0])
        sns.lineplot(data=history, x='epoch', y='-log p(x|z)', hue='type', ax=ax[1])

        history = pd.DataFrame({'epoch': np.arange(len(train_avg)),
                                'loss': train_avg,
                                '-log p(x|z)': recon_avg,
                                'log p(z)': dist_avg,
                                # 'task': task_avg,
                                'type': 'train'})
        temp = pd.DataFrame({'epoch': np.arange(len(train_avg)),
                                'loss': val_avg,
                                '-log p(x|z)': recon_val_avg,
                                'log p(z)': dist_val_avg,
                                # 'task': task_val_avg,
                                'type': 'test'}, index=[x+len(train_avg) for x in np.arange(len(train_avg))])
        history = pd.concat([history, temp])

        scatterplot_sizes = (6, 6, 40)
        fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(20, 5))
        sns.lineplot(data=history, x='epoch', y='loss', hue='type', ax=ax[0])
        sns.lineplot(data=history, x='epoch', y='-log p(x|z)', hue='type', ax=ax[1])
        sns.lineplot(data=history, x='epoch', y='log p(z)', hue='type', ax=ax[2])

    pca = PCA(n_components=2)
    pca_data = pd.DataFrame(data=rep.z.cpu().detach().numpy())
    pca_data['type'] = 'representation'
    pca_data['size'] = scatterplot_sizes[0]
    if gmm is not None:
        sampled = gmm.sample(500)
        temp = pd.DataFrame(data=sampled.cpu().detach().numpy())
        temp['type'] = 'gmm'
        temp['size'] = scatterplot_sizes[1]
        pca_data = pd.concat([pca_data, temp])
        if len(gmm.mean.cpu().detach().numpy().shape) > 1:
            temp = pd.DataFrame(data=gmm.mean.cpu().detach().numpy())
            temp['type'] = 'gmm_mean'
            temp['size'] = scatterplot_sizes[2]
            pca_data = pd.concat([pca_data, temp])
        pca.fit(pca_data.iloc[:, :-2])
        projected = pca.fit_transform(pca_data.iloc[:, :-2])
        projected_norm = (projected - np.mean(projected.flatten())) / np.std(projected.flatten())
        pca_data2 = pd.DataFrame(data=projected_norm, columns=['PC1', 'PC2'])
        pca_data2['type'] = pca_data['type'].values
        pca_data2['size'] = pca_data['size'].values
        sns.scatterplot(x='PC1', y='PC2', data=pca_data2, hue='type', size='size', ax=ax[3], sizes=(scatterplot_sizes[0], scatterplot_sizes[2]))
        ax[3].legend(bbox_to_anchor=(1.05, 1.), loc='upper left')
    else:
        pca.fit(pca_data.iloc[:, :-2])
        projected = pca.fit_transform(pca_data.iloc[:, :-2])
        projected_norm = (projected - np.mean(projected.flatten())) / np.std(projected.flatten())
        pca_data2 = pd.DataFrame(data=projected_norm, columns=['PC1', 'PC2'])
        pca_data2['type'] = pca_data['type'].values
        pca_data2['size'] = pca_data['size'].values
        sns.scatterplot(x='PC1', y='PC2', data=pca_data2, hue='type', size='size', ax=ax[3], sizes=(scatterplot_sizes[0], scatterplot_sizes[2]))
        ax[3].legend(bbox_to_anchor=(1.05, 1.), loc='upper left')
    plt.tight_layout()
    plt.show()

    model.load_state_dict(best_model)
    print("Best epoch:", best_epoch, "val loss:", best_val_loss)

    return model, history


def train_nbvae(model, opt, train_loader, val_loader, num_epochs, beta=1, warmup_epochs=15, patience=3, dev="cpu", print_epoch=1):
    loss_history = {'train_loss': [], 'recon_loss': [], 'kl_loss': [], 'val_loss': []}
    
    best_model = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    best_epoch = -1
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        running_recon_loss = 0.0
        running_kl_loss = 0.0
        kl_weight = min(beta, 1e-5 * warmup_epochs + (1 - 1e-5 * warmup_epochs) * (epoch / warmup_epochs))

        for _, data, data_max, _, conditional_vars in train_loader:
            data, data_max, conditional_vars = data.to(dev), data_max.to(dev), conditional_vars.to(dev)

            opt.zero_grad()
            x_hat, mu, log_var = model(data, conditional_vars if model.n_conditional_vars > 0 else None)
            recon_loss, kl_div = model.loss(data, x_hat, mu, log_var, data_max)
            loss = recon_loss + kl_weight * kl_div
            loss.backward()
            opt.step()

            running_loss += loss.item()
            running_recon_loss += recon_loss.item()
            running_kl_loss += kl_div.item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_recon_loss = running_recon_loss / len(train_loader.dataset)
        epoch_kl_loss = running_kl_loss / len(train_loader.dataset)
        loss_history['train_loss'].append(epoch_loss)
        loss_history['recon_loss'].append(epoch_recon_loss)
        loss_history['kl_loss'].append(epoch_kl_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for _, data, data_max, _, conditional_vars in val_loader:
                data, data_max, conditional_vars = data.to(dev), data_max.to(dev), conditional_vars.to(dev)
                x_hat, mu, log_var = model(data, conditional_vars if model.n_conditional_vars > 0 else None)
                recon_loss, kl_div = model.loss(data, x_hat, mu, log_var, data_max)
                loss = recon_loss + kl_weight * kl_div
                val_loss += loss.item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        loss_history['val_loss'].append(epoch_val_loss)

        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping")
            break

        if epoch % print_epoch == 0:
            print(f"Epoch: {epoch}, Loss: {epoch_loss}, Recon Loss: {epoch_recon_loss}, KL Loss: {epoch_kl_loss}, Val Loss: {epoch_val_loss}")
    
    model.load_state_dict(best_model)
    print("Best epoch:", best_epoch, "val loss:", best_val_loss)
    return model, loss_history


scatterplot_sizes = [2,2,20]
def plot_negbino(model, tdata, r, e, gmm, t='nb', scaling='library', labels=None, library=None, label_cols=1, device='cpu'):
    # move to cpu
    mod = model.cpu()
    r = r.cpu()
    gmm = gmm.cpu()

    print('epoch: {}'.format(e))
    if library is None:
        if scaling == 'library':
            library = torch.max(tdata, dim=-1).values.unsqueeze(1)
        elif scaling == 'mean':
            library = torch.mean(tdata, dim=-1).unsqueeze(1)

    if t == 'nb':
        y = mod(r.z)
        y_scaled = (mod(r.z) * library).cpu()
    else:
        y = mod(r.z)
        y_scaled = y.cpu()
    
    fig, ax = plt.subplots(ncols=4,nrows=2,figsize=(20,6))
    plt.subplots_adjust(left=None, bottom=0.5, right=None, top=1.5, wspace=None, hspace=None)
    
    # datasets for first row of plots
    plotdata = pd.DataFrame({'data': tdata[:,0] + 1,
                            'type': 'original',
                            'gene': '1'})
    temp = pd.DataFrame({'data': y_scaled[:,0].detach().flatten() + 1,
                        'type': 'reconstruction',
                        'gene': '1'}, index=[x+library.shape[0] for x in np.arange(library.shape[0])])
    plotdata = pd.concat([plotdata, temp])
    temp = pd.DataFrame({'data': tdata[:,1].flatten() + 1,
                            'type': 'original',
                            'gene': '2'}, index=[x+2*library.shape[0] for x in np.arange(library.shape[0])])
    plotdata = pd.concat([plotdata, temp])
    temp = pd.DataFrame({'data': y_scaled[:,1].detach().flatten() + 1,
                        'type': 'reconstruction',
                        'gene': '2'}, index=[x+3*library.shape[0] for x in np.arange(library.shape[0])])
    plotdata = pd.concat([plotdata, temp])

    plotdata2 = pd.DataFrame({'original': tdata[:,0] + 1,
                              'reconstructed': y_scaled[:,0].detach().flatten() + 1,
                            'neg_log_prob': mod.nb.loss(tdata, library, y)[:,0].detach().flatten(),
                            'gene': '1'})
    temp = pd.DataFrame({'original': tdata[:,1] + 1,
                        'reconstructed': y_scaled[:,1].detach().flatten() + 1,
                            'neg_log_prob': mod.nb.loss(tdata, library, y)[:,1].detach().flatten(),
                            'gene': '2'}, index=[x+library.shape[0] for x in np.arange(library.shape[0])])
    plotdata2 = pd.concat([plotdata2, temp])
    
    # gene reconstruction plots
    sns.histplot(data=plotdata[(plotdata['type'] != 'neg_log_prob') & (plotdata['gene'] == '1')], 
                 x='data', hue='type', ax=ax[0][0], log_scale=(True, True))
    ax[0][0].set_title('gene 1')
    sns.histplot(data=plotdata[(plotdata['type'] != 'neg_log_prob') & (plotdata['gene'] == '2')], 
                 x='data', hue='type', ax=ax[0][1], log_scale=(True, True))
    ax[0][1].set_title('gene 2')
    sns.scatterplot(data=plotdata2[plotdata2['gene'] == '1'], x='original', y='reconstructed', hue='neg_log_prob', ax=ax[0][2])
    r1 = np.round(torch.exp(mod.nb.log_r[0,0]).detach().numpy(), 4)
    ax[0][2].set_title('gene 1 (r='+str(r1)+')')
    ax[0][2].set_yscale('log')
    ax[0][2].set_xscale('log')
    sns.scatterplot(data=plotdata2[plotdata2['gene'] == '2'], x='original', y='reconstructed', hue='neg_log_prob', ax=ax[0][3])
    r2 = np.round(torch.exp(mod.nb.log_r[0,1]).detach().numpy(), 4)
    ax[0][3].set_title('gene 2 (r='+str(r2)+')')
    ax[0][3].set_yscale('log')
    ax[0][3].set_xscale('log')
    
    # pca plots with many conditionals
    if r.nrep > 2:
      pca = PCA(n_components=2)
      if gmm is not None:
        sampled = gmm.sample(500)
        pca_data = pd.DataFrame(data=r.z.cpu().detach().numpy())
        pca_data['type'] = 'representation'
        pca_data['size'] = scatterplot_sizes[0]
        if labels is not None:
          if len(labels) == len(pca_data):
              pca_data['label'] = labels
          else:
              pca_data['label'] = labels[:len(pca_data)]
        temp = pd.DataFrame(data=sampled.cpu().detach().numpy())
        temp['type'] = 'gmm'
        temp['size'] = scatterplot_sizes[1]
        if labels is not None:
            temp['label'] = 'gmm'
        pca_data = pd.concat([pca_data, temp])
        if len(gmm.mean.cpu().detach().numpy().shape) > 1:
            temp = pd.DataFrame(data=gmm.mean.cpu().detach().numpy())
            temp['type'] = 'gmm_mean'
            temp['size'] = scatterplot_sizes[2]
            if labels is not None:
                temp['label'] = 'gmm_mean'
            pca_data = pd.concat([pca_data, temp])
        if labels is not None:
            pca.fit(pca_data.iloc[:,:-3])
            projected = pca.fit_transform(pca_data.iloc[:,:-3])
        else:
            pca.fit(pca_data.iloc[:,:-2])
            projected = pca.fit_transform(pca_data.iloc[:,:-2])
        projected_norm = (projected - np.mean(projected.flatten())) / np.std(projected.flatten())
        pca_data2 = pd.DataFrame(data=projected_norm, columns=['PC1','PC2'])
        pca_data2['type'] = pca_data['type'].values
        pca_data2['size'] = pca_data['size'].values
        pca_data2['label'] = pca_data['label'].values
        sns.scatterplot(x='PC1', y='PC2', data= pca_data2, hue='type', size='size', ax=ax[1][0], sizes=(4,20), palette=['lightblue','orange','black'])
        ax[1][0].legend(bbox_to_anchor=(1.05, 1.), loc='upper left')
        fig.delaxes(ax[1][1])
        fig.delaxes(ax[1][3])
        if labels is not None:
            sns.scatterplot(x='PC1', y='PC2', data= pca_data2[pca_data2['type'] == 'representation'], hue='label', ax=ax[1][2], s=6)
            ax[1][2].legend(bbox_to_anchor=(1.05, 1.), loc='upper left',ncol=label_cols)
      else:
        pca_data = pd.DataFrame(data=r.z.cpu().detach().numpy())
        pca_data['type'] = 'representation'
        pca.fit(pca_data.iloc[:,:-1])
        projected = pca.fit_transform(pca_data.iloc[:,:-1])
        projected_norm = (projected - np.mean(projected.flatten())) / np.std(projected.flatten())
        pca_data2 = pd.DataFrame(data=projected_norm, columns=['PC1','PC2'])
        pca_data2['type'] = pca_data['type'].values
        if labels is not None:
            if len(labels) == len(pca_data):
                sns.scatterplot(x='PC1', y='PC2', data= pca_data2, hue=labels, ax=ax[1][0], s=6)
            else:
                sns.scatterplot(x='PC1', y='PC2', data= pca_data2, hue=labels[:len(pca_data)], ax=ax[1][0], s=6)
        else:
            sns.scatterplot(x='PC1', y='PC2', data= pca_data2, hue='type', ax=ax[4], sizes=(4,20))
        ax[1][0].legend(bbox_to_anchor=(1.05, 1.), loc='upper left',ncol=label_cols)
        fig.delaxes(ax[1][1])
        fig.delaxes(ax[1][2])
        fig.delaxes(ax[1][3])
    else:
      if gmm is not None:
        sampled = gmm.sample(500)
        pca_data = pd.DataFrame(data=r.z.cpu().detach().numpy(), columns=['D1','D2'])
        pca_data['type'] = 'representation'
        pca_data['size'] = scatterplot_sizes[0]
        if labels is not None:
          if len(labels) == len(pca_data):
              pca_data['label'] = labels
          else:
              pca_data['label'] = labels[:len(pca_data)]
        temp = pd.DataFrame(data=sampled.cpu().detach().numpy(), columns=['D1','D2'])
        temp['type'] = 'gmm'
        temp['size'] = scatterplot_sizes[1]
        if labels is not None:
            temp['label'] = 'gmm'
        pca_data = pd.concat([pca_data, temp])
        if len(gmm.mean.cpu().detach().numpy().shape) > 1:
            temp = pd.DataFrame(data=gmm.mean.cpu().detach().numpy(), columns=['D1','D2'])
            temp['type'] = 'gmm_mean'
            temp['size'] = scatterplot_sizes[2]
            if labels is not None:
                temp['label'] = 'gmm_mean'
            pca_data = pd.concat([pca_data, temp])
        sns.scatterplot(x='D1', y='D2', data= pca_data, hue='type', size='size', ax=ax[1][0], sizes=(4,20), palette=['lightblue','orange','black'])
        ax[1][0].legend(bbox_to_anchor=(1.05, 1.), loc='upper left')
        fig.delaxes(ax[1][1])
        fig.delaxes(ax[1][3])
        if labels is not None:
            sns.scatterplot(x='D1', y='D2', data= pca_data[pca_data['type'] == 'representation'], hue='label', ax=ax[1][2], s=6)
            ax[1][2].legend(bbox_to_anchor=(1.05, 1.), loc='upper left',ncol=label_cols)
      else:
        pca_data = pd.DataFrame(data=r.z.cpu().detach().numpy(), columns=['D1','D2'])
        pca_data['type'] = 'representation'
        if labels is not None:
            if len(labels) == len(pca_data):
                sns.scatterplot(x='D1', y='D2', data= pca_data, hue=labels, ax=ax[1][0], s=6)
            else:
                sns.scatterplot(x='D1', y='D2', data= pca_data, hue=labels[:len(pca_data)], ax=ax[1][0], s=6)
        else:
            sns.scatterplot(x='D1', y='D2', data= pca_data, hue='type', ax=ax[4], sizes=(4,20))
        ax[1][0].legend(bbox_to_anchor=(1.05, 1.), loc='upper left',ncol=label_cols)
        fig.delaxes(ax[1][1])
        fig.delaxes(ax[1][2])
        fig.delaxes(ax[1][3])
    
    plt.show()
