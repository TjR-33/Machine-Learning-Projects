import torch
from torch import nn, optim, save, load
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from torcheval.metrics.functional import binary_f1_score, binary_accuracy
from tqdm.auto import tqdm
import os 

class PlantHealthClassifier(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.model = nn.Sequential(

            nn.Conv2d(input_channels, 32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(18432 , 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.model(x)

class PlantHealthClassifier2(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.model = nn.Sequential(

            nn.Conv2d(input_channels,64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(73728 , 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 1)
        )

    def forward(self,X):
        return self.model(X)

def _train_model_with_early_stopping(model, train_loader, val_loader, optimizer, scheduler,  n=2, p=7, epochs = 50, loss_fn = nn.BCEWithLogitsLoss(),device='cpu'):
    """
    Trains a PyTorch model with early stopping.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    train_loader : torch.utils.data.DataLoader
        The DataLoader for the training data.
    val_loader : torch.utils.data.DataLoader
        The DataLoader for the validation data.
    optimizer : torch.optim.Optimizer
        The optimizer to be used for training.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        The learning rate scheduler.
    n : int, optional
        The frequency of validation checks for early stopping.
    p : int, optional
        The number of epochs with no improvement after which training will be stopped.
    epochs : int, optional
        The maximum number of epochs for training.
    loss_fn : torch.nn.modules.loss._Loss, optional
        The loss function.
    device : str, optional
        The device to which the data and model are sent ('cpu' or 'cuda').

    Returns
    -------
    i_best : int
        The epoch with the best validation loss.
    v_best : float
        The best validation loss.
    val_f1_score : float
        The F1 score on the validation set at the best epoch.
    val_accuracy : float
        The accuracy on the validation set at the best epoch.
    epoch_metrics : dict
        A dictionary containing the epoch-wise metrics.
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if loss_fn is None:
        loss_fn=nn.BCEWithLogitsLoss()

    # Initialize early stopping parameters
    theta_best = None
    i_best = 0
    j = 0
    v_best = float('inf')
    # Initialize epoch-wise metrics dictionary
    epoch_metrics = {
        'epoch_count': [],
        'train_loss_values': [],
        'val_loss_values': [],
        'train_f1_values': [],
        'val_f1_values': [],
        'train_accuracy_values': [],
        'val_accuracy_values': []
    }

    print("Training model...")

    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch + 1}\n------")

        train_loss = 0.0
        train_f1_score = 0.0
        train_accuracy = 0.0

        # Training phase

        # Load all batches first them iterate

        for batch, (X, y) in enumerate(train_loader):
            model.train()
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            y_logits = model(X)

            loss = loss_fn(y_logits, y.unsqueeze(1).float())
            train_f1_score += binary_f1_score(torch.round(torch.sigmoid(y_logits)).squeeze(), y).item()
            train_accuracy += binary_accuracy(torch.round(torch.sigmoid(y_logits)).squeeze(), y).item()

            train_loss += loss.item()

            loss.backward()

            optimizer.step()

            # Print progress
            if batch % 4 == 0:
                print(f"Looked at {batch * len(X)}/{len(train_loader.dataset)} samples.")
        
        if scheduler is not None:
            scheduler.step()

        # Compute average training loss
        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)
        train_f1_score /= len(train_loader)


        print(f"Looked at {len(train_loader.dataset)}/{len(train_loader.dataset)} samples.")

        # Validation phase
        val_loss = 0.0
        val_f1_score = 0.0
        val_accuracy = 0.0

        model.eval()
        with torch.inference_mode():
            for X_test, y_test in val_loader:
                X_test = X_test.to(device)
                y_test = y_test.to(device)

                y_test_logits = model(X_test)
                val_loss += loss_fn(y_test_logits, y_test.unsqueeze(1).float()).item()
                val_accuracy += binary_accuracy(torch.round(torch.sigmoid(y_test_logits)).squeeze(), y_test).item()
                y_test_logits_rounded = torch.round(torch.sigmoid(y_test_logits)).squeeze()
                if len(y_test_logits_rounded.shape) == 0:  # if it's a scalar
                    y_test_logits_rounded = y_test_logits_rounded.unsqueeze(0)  # add a dimension
                val_f1_score += binary_f1_score(y_test_logits_rounded, y_test).item()

            val_loss /= len(val_loader)
            val_accuracy /= len(val_loader)
            val_f1_score /= len(val_loader)

        # Update epoch-wise metrics dictionary
        epoch_metrics['epoch_count'].append(epoch)
        epoch_metrics['train_loss_values'].append(train_loss)
        epoch_metrics['val_loss_values'].append(val_loss)
        epoch_metrics['train_f1_values'].append(train_f1_score)
        epoch_metrics['val_f1_values'].append(val_f1_score)
        epoch_metrics['train_accuracy_values'].append(train_accuracy)
        epoch_metrics['val_accuracy_values'].append(val_accuracy)


        # Early stopping check
        if epoch % n == 0:
            if val_loss < v_best:
                j = 0
                theta_best = model.state_dict()
                i_best = epoch
                v_best = val_loss
            else:
                j += 1
                if j >= p:
                    print("Validation error did not improve for the last", p, "evaluations. Early stopping.")
                    model.load_state_dict(theta_best)
                    print("Model restored to the state of epoch", i_best+1)
                    print("Best Validation loss:", v_best)
                    print("Best Validation F1-Score:", val_f1_score)
                    print("Best Validation accuracy:", val_accuracy)
                    break

        print(f"Training loss: {train_loss:.4f} | Training F1-Score: {train_f1_score:.4f} | Training Accuracy: {train_accuracy} |\nValidation Error: {val_loss:.4f} | Validation F1-Score: {val_f1_score:.4f} | Validation Accuracy: {val_accuracy}\n")
        if device == "cuda":
            torch.cuda.empty_cache()
        
    return i_best,v_best,val_f1_score, val_accuracy, epoch_metrics 

class EnsembleModel:
    """
    A class used to represent an Ensemble Model for classification tasks.

    ...

    Attributes
    ----------
    model : nn.Module
        the base model to be used in the ensemble
    input_channels : int
        the number of input channels for the base model
    ensemble : nn.ModuleList
        a list of models in the ensemble
    device : str
        the device to which the model is sent
    num_models : int
        the number of models in the ensemble
    weighted_voting : bool
        whether to use weighted voting in the ensemble
    voting_weights : torch.Tensor
        the weights used for voting in the ensemble

    Methods
    -------
    bagging(num_models, train_dataset, test_dataset, batch_size, num_epochs, optimizer_factory=None, scheduler_factory=None, lr=0.001, lambda_lr=lambda epoch: 0.95 ** epoch, n=2, p=7, get_training_results=False, save_dir=None):
        Trains the ensemble model using bagging and tests it on the test dataset.
    voting(ensemble_preds, weights):
        Performs weighted voting on the predictions of the ensemble models.
    predict(test_dataset):
        Predicts the labels for the test dataset using the ensemble model.
    _generate_bootstrapped_dataloaders(dataset, batch_size, num_bootstraps):
        Generates bootstrapped dataloaders for training the ensemble models.
    save(save_dir='ensemble_model', save_state_dict=True):
        Saves the ensemble models and their voting weights to the specified directory.
    save_individual_model(save_dir='ensemble_model', model=None, filename='model.pth'):
        Saves an individual model to the specified directory.
    load(save_dir='ensemble_model', load_state_dict=True):
        Loads the ensemble models and their voting weights from the specified directory.
    """
    def __init__(self,model, input_channels, device, weighted_voting=False):
        self.model = model
        self.input_channels = input_channels
        self.ensemble = None
        self.num_models = None
        self.device = device
        self.weighted_voting = weighted_voting
        self.voting_weights = None

    def bagging(self,
                num_models,
                train_dataset, test_dataset, 
                batch_size, 
                num_epochs, 
                optimizer_factory = None,
                scheduler_factory = None,
                lr=0.001, lambda_lr = lambda epoch: 0.95 ** epoch,
                n=2, p=7,
                get_training_results = False,
                save_dir = None):

        self.ensemble = nn.ModuleList([self.model(self.input_channels).to(self.device) for _ in range(num_models)])
        self.num_models = num_models
        self.voting_weights = torch.ones(num_models) / num_models

        bootstrapped_dataloaders, OOB_dataloaders = self._generate_bootstrapped_dataloaders(train_dataset, batch_size, len(self.ensemble))
        # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        OOB_error_list = []
        acc_list = []
        f1_list = []
        best_num_epochs_list = []
        epoch_metrics_list = [] if get_training_results else None

        for i, model in enumerate(self.ensemble):
            optimizer = optimizer_factory(model.parameters(), lr) if optimizer_factory is not None else optim.Adam(model.parameters(), lr=lr)
            scheduler = scheduler_factory(optimizer, lambda_lr) if scheduler_factory is not None else None 
            loss_fn = nn.BCEWithLogitsLoss()

            print(f"Training model {i + 1}/{len(self.ensemble)}...")
            print("We are using OOB observations for validation and early stopping.")
            if self.device == "cuda":
                torch.cuda.empty_cache()
            best_num_epochs,OOB_error,f1,acc,epoch_metrics =_train_model_with_early_stopping(model=model, 
                                                                 train_loader=bootstrapped_dataloaders[i], val_loader=OOB_dataloaders[i], 
                                                                 optimizer=optimizer,scheduler= scheduler, loss_fn=loss_fn, epochs=num_epochs, n=n, p=p, device= self.device)
            
            if save_dir is not None:
                self.save_individual_model(save_dir=save_dir, model=model, filname=f'model_{i}.pth')
            
            OOB_error_list.append(OOB_error)
            acc_list.append(acc)
            f1_list.append(f1)
            best_num_epochs_list.append(best_num_epochs)
            if get_training_results:
                epoch_metrics_list.append(epoch_metrics)
        if self.weighted_voting:
            self.voting_weights = 1 / torch.tensor(OOB_error_list)
        else:
            self.voting_weights = torch.ones(self.num_models) / self.num_models

        # Testing the model on test_loader

        X_test = []
        y_test = []

        for i in range(len(test_dataset)):
            X,y = test_dataset.__getitem__(i)
            X=X.to(self.device)
            y = torch.tensor(y).to(self.device)
            X_test.append(X)
            y_test.append(y)
            del X,y
        X_test = torch.stack(X_test).to(self.device)
        y_test = torch.stack(y_test).to(self.device).squeeze()
        
        y_pred = self.predict(test_dataset=X_test).to(self.device).squeeze()

        del X_test

        test_loss = nn.BCELoss()(y_pred.float(), y_test.float())/len(y_test)
        test_f1 = binary_f1_score(y_pred, y_test).item()
        test_accuracy = binary_accuracy(y_pred, y_test).item()
    
        print(f"\nTest loss: {test_loss:.4f} | Test F1-Score: {test_f1:.4f} | Test Accuracy: {test_accuracy}")
        return test_loss, test_f1, test_accuracy, best_num_epochs_list, OOB_error_list, acc_list, f1_list, epoch_metrics_list

    def voting(self, ensemble_preds, weights):
        # ensemble_preds -> n_samples, num_models

        preds =[]
        for i in range(len(ensemble_preds)):
            ensemble_pred = ensemble_preds[i]
            # ensemble_pred -> num_models
            score_1 = 0
            score_0 = 0
            for j in range(len(ensemble_pred)):
                if ensemble_pred[j] == 1:
                    score_1 += weights[j]
                else:
                    score_0 += weights[j]
            
            if score_1 > score_0:
                preds.append(1)
            else:
                preds.append(0)

        return torch.tensor(preds)             
        # return torch.mean(ensemble_preds * weights.unsqueeze(1), dim=0)

    def predict(self,  test_dataset):
        # test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        # X -> n_samples, n_channels, height, width
        # ensemble_preds -> n_samples, num_models
        ensemble_preds= []
        X = test_dataset
        if len(X.shape) != 4:
            print("X is not a 4D tensor or array....\nReshaping it to 4D tensor")
            X = X.unsqueeze(0)

        X = torch.tensor(test_dataset).to(self.device) if isinstance(test_dataset, np.ndarray) else test_dataset

        for model in self.ensemble:
            model.eval()
            with torch.inference_mode():
                    logits = model(X)
                    preds = torch.round(torch.sigmoid(logits))
                    # preds -> n_samples
                    ensemble_preds.append(preds)

        ensemble_preds=torch.stack(ensemble_preds, dim=1).squeeze(-1)
        return self.voting(ensemble_preds, self.voting_weights)
        # return None

    def _generate_bootstrapped_dataloaders(self, dataset, batch_size, num_bootstraps):
        bootstrapped_dataloaders = []
        OOB_dataloaders = []
        dataset_size = len(dataset)

        for _ in range(num_bootstraps):
            indices = np.random.choice(dataset_size, dataset_size, replace=True)
            bootstrapped_dataset = Subset(dataset, indices)
            # make OOB_dataset by inverting the indices
            OOB_indices = np.setdiff1d(np.arange(dataset_size), indices)
            #make sure indices are unique
            OOB_indices = np.unique(OOB_indices)
            OOB_dataset = Subset(dataset, OOB_indices)
            bootstrapped_dataloader = DataLoader(bootstrapped_dataset, batch_size=batch_size, shuffle=True)
            OOB_dataloader = DataLoader(OOB_dataset, batch_size=batch_size, shuffle=True)
            bootstrapped_dataloaders.append(bootstrapped_dataloader)
            OOB_dataloaders.append(OOB_dataloader)
        return bootstrapped_dataloaders, OOB_dataloaders
    
    def save(self,save_dir='ensemble_model', save_state_dict=True):
        print("Saving model...")
        os.mkdir(save_dir) if not os.path.exists(save_dir) else None
        for i,model in enumerate(self.ensemble):
            if f'{save_dir}/model_{i}.pth' in os.listdir(save_dir):
                print(f"File {save_dir}/model_{i}.pth already exists. Overwriting it.")
            if save_state_dict:
                save(model.state_dict(), f'{save_dir}/model_{i}.pth')
            else:
                save(model, f'{save_dir}/model_{i}.pth')
        print(f"Saved {self.num_models} models.")
        print("Saving Voting weights...")
        save(self.voting_weights, f'{save_dir}/voting_weights.pth')

    def save_individual_model(self,save_dir='ensemble_model',model=None,filname='model.pth'):
        os.mkdir(save_dir) if not os.path.exists(save_dir) else None
        save(model.state_dict(), f'{save_dir}/{filname}')
        print(f"Saved model to {save_dir}/{filname}")

    def load(self, save_dir='ensemble_model', load_state_dict=True):
        print("Loading model...")
        if not os.path.exists(save_dir):
            raise FileNotFoundError(f"Directory {save_dir} does not exist.")
        self.ensemble = nn.ModuleList()
        # Find all .pth files
        files = [f for f in os.listdir(save_dir) if f.endswith('.pth')]
        # remove voting_weights.pth from files
        files = [f for f in files if f != 'voting_weights.pth']
        if files == []:
            raise FileNotFoundError(f"No .pth files found in {save_dir}.")
        else:
            for file in files:
                if load_state_dict:
                    # Correctly instantiate a new model for each file
                    model= self.model(self.input_channels)  
                    model.to(self.device)
                    model.load_state_dict(load(f'{save_dir}/{file}'))
                    self.ensemble.append(model)
                else:
                    model = torch.load(f'{save_dir}/{file}')
                    model.to(self.device)
                    self.ensemble.append(model)
        self.num_models = len(self.ensemble)
        print(f"Loaded {self.num_models} models.")
        if self.weighted_voting:
            if "voting_weights.pth" in os.listdir(save_dir):
                print("Loading Voting weights...")
                self.voting_weights = torch.load(f'{save_dir}/voting_weights.pth')
                print(f"Loaded Voting weights. {self.voting_weights}")
            else:
                raise FileNotFoundError("voting_weights.pth not found in the directory.")
        else:
            self.voting_weights = torch.ones(self.num_models) / self.num_models
            
