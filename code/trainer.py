import os
import time
import tqdm
import pandas as pd
from copy import deepcopy
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn import metrics

def print_metrics_binary(y_true, predictions, verbose=False):
    predictions = np.array(predictions)
    # if len(predictions.shape) == 1:
    #     predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))

    # cf = metrics.confusion_matrix(y_true, predictions.argmax(axis=1))
    # if verbose:
    #     print("confusion matrix:")
    #     print(cf)
    # cf = cf.astype(np.float32)

    # acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
    # prec0 = cf[0][0] / (cf[0][0] + cf[1][0])
    # prec1 = cf[1][1] / (cf[1][1] + cf[0][1])
    # rec0 = cf[0][0] / (cf[0][0] + cf[0][1])
    # rec1 = cf[1][1] / (cf[1][1] + cf[1][0])
    auroc = metrics.roc_auc_score(y_true, predictions)

    (precisions, recalls, thresholds) = metrics.precision_recall_curve(y_true, predictions)
    auprc = metrics.auc(recalls, precisions)
    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])

    if verbose:
        # print("accuracy = {}".format(acc))
        # print("precision class 0 = {}".format(prec0))
        # print("precision class 1 = {}".format(prec1))
        # print("recall class 0 = {}".format(rec0))
        # print("recall class 1 = {}".format(rec1))
        print("AUC of ROC = {}".format(auroc))
        print("AUC of PRC = {}".format(auprc))
        print("min(+P, Se) = {}".format(minpse))

    return {
      # "acc": acc,
      # "prec0": prec0,
      # "prec1": prec1,
      # "rec0": rec0,
      # "rec1": rec1,
      "auroc": auroc,
      "auprc": auprc,
      "minpse": minpse
      }


def print_metrics_multilabel(y_true, predictions, verbose=False):
    y_true = np.array(y_true)
    predictions = np.array(predictions)

    # auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)
    ave_auc_micro = metrics.roc_auc_score(y_true, predictions,
                                          average="micro")
    ave_auc_macro = metrics.roc_auc_score(y_true, predictions,
                                          average="macro")
    ave_auc_weighted = metrics.roc_auc_score(y_true, predictions,
                                             average="weighted")

    if verbose:
        # print("ROC AUC scores for labels:", auc_scores)
        print("ave_auc_micro = {}".format(ave_auc_micro))
        print("ave_auc_macro = {}".format(ave_auc_macro))
        print("ave_auc_weighted = {}".format(ave_auc_weighted))

    return {
      #  "auc_scores": auc_scores,
            "ave_auc_micro": ave_auc_micro,
            "ave_auc_macro": ave_auc_macro,
            "ave_auc_weighted": ave_auc_weighted}
class CustomBins:
    inf = 1e18
    bins = [(-inf, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 14), (14, +inf)]
    nbins = len(bins)
    means = [11.450379, 35.070846, 59.206531, 83.382723, 107.487817,
             131.579534, 155.643957, 179.660558, 254.306624, 585.325890]


def get_bin_custom(x, nbins, one_hot=False):
    for i in range(nbins):
        a = CustomBins.bins[i][0] * 24.0
        b = CustomBins.bins[i][1] * 24.0
        if a <= x < b:
            if one_hot:
                ret = np.zeros((CustomBins.nbins,))
                ret[i] = 1
                return ret
            return i
    return None

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 0.1))) * 100

def print_metrics_regression(y_true, predictions, verbose=False):
    predictions = np.array(predictions)
    prediction_bins = predictions
    # predictions = np.array(predictions)
    # predictions = np.maximum(predictions, 0).flatten()
    y_true = np.array(y_true)
    y_true_bins = y_true
    # y_true_bins = [get_bin_custom(x, CustomBins.nbins) for x in y_true]
    # prediction_bins = [get_bin_custom(x, CustomBins.nbins) for x in predictions]
    # cf = metrics.confusion_matrix(y_true_bins, prediction_bins)
    if verbose:
        print("Custom bins confusion matrix:")
        # print(cf)

    kappa = metrics.cohen_kappa_score(y_true_bins, prediction_bins,
                                      weights='linear')
    mad = metrics.mean_absolute_error(y_true, predictions)
    mse = metrics.mean_squared_error(y_true, predictions)
    mape = mean_absolute_percentage_error(y_true, predictions)

    if verbose:
        print("Mean absolute deviation (MAD) = {}".format(mad))
        print("Mean squared error (MSE) = {}".format(mse))
        print("Mean absolute percentage error (MAPE) = {}".format(mape))
        print("Cohen kappa score = {}".format(kappa))

    return {"mad": mad,
            "mse": mse,
            "mape": mape,
            "kappa": kappa}

def print_metrics_custom_bins(y_true, predictions, verbose=1):
    return print_metrics_regression(y_true, predictions, verbose)

class NeuralNetworkClassifier:
    """
    | NeuralNetworkClassifier depend on `Comet-ML <https://www.comet.ml/>`_ .
    | You have to create a project on your workspace of Comet, if you use this class.
    |
    | example

    ---------------------
    1st, Write your code.
    ---------------------
    ::

        # code.py
        from comet_ml import Experiment
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from SAnD.utils.trainer import NeuralNetworkClassifier

        class Network(nn.Module):
           def __init__(self):
               super(Network ,self).__init__()
               ...
           def forward(self, x):
               ...

        optimizer_config = {"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-08}
        comet_config = {}

        train_val_loader = {
           "train": train_loader,
           "val": val_loader
        }
        test_loader = DataLoader(test_ds, batch_size)

        clf = NeuralNetworkClassifier(
                Network(), nn.CrossEntropyLoss(),
                optim.Adam, optimizer_config, Experiment()
            )

        clf.experiment_tag = "experiment_tag"
        clf.num_classes = 3
        clf.fit(train_val_loader, epochs=10)
        clf.evaluate(test_loader)
        lf.confusion_matrix(test_ds)
        clf.save_weights("save_params_test/")

    ----------------------------
    2nd, Run code on your shell.
    ----------------------------
    | You need to define 2 environment variables.
    | :code:`COMET_API_KEY` & :code:`COMET_PROJECT_NAME`

    On Unix-like system, you can define them like this and execute code.
    ::

        export COMET_API_KEY="YOUR-API-KEY"
        export COMET_PROJECT_NAME="YOUR-PROJECT-NAME"
        user@user$ python code.py

    -------------------------------------------
    3rd, check logs on your workspace of comet.
    -------------------------------------------
    Just access your `Comet-ML <https://www.comet.ml/>`_ Project page.

    ^^^^^
    Note,
    ^^^^^

    Execute this command on your shell, ::

        export COMET_DISABLE_AUTO_LOGGING=1

    If the following error occurs. ::

        ImportError: You must import Comet before these modules: torch

    """
    def __init__(self, model, criterion, optimizer, optimizer_config: dict) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer(self.model.parameters(), **optimizer_config)
        self.criterion = criterion
        # self.experiment = experiment

        self.hyper_params = optimizer_config
        self._start_epoch = 0
        self.hyper_params["epochs"] = self._start_epoch
        self.__num_classes = None
        self._is_parallel = False

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            self._is_parallel = True

            notice = "Running on {} GPUs.".format(torch.cuda.device_count())
            print("\033[33m" + notice + "\033[0m")

    def fit(self, loader: Dict[str, DataLoader], epochs: int, checkpoint_path: str = None, validation: bool = True) -> None:
        """
        | The method of training your PyTorch Model.
        | With the assumption, This method use for training network for classification.

        ::

            train_ds = Subset(train_val_ds, train_index)
            val_ds = Subset(train_val_ds, val_index)

            train_val_loader = {
                "train": DataLoader(train_ds, batch_size),
                "val": DataLoader(val_ds, batch_size)
            }

            clf = NeuralNetworkClassifier(
                    Network(), nn.CrossEntropyLoss(),
                    optim.Adam, optimizer_config, experiment
                )
            clf.fit(train_val_loader, epochs=10)


        :param loader: Dictionary which contains Data Loaders for training and validation.: dict{DataLoader, DataLoader}
        :param epochs: The number of epochs: int
        :param checkpoint_path: str
        :param validation:
        :return: None
        """
        len_of_train_dataset = len(loader["train"].dataset)
        epochs = epochs + self._start_epoch

        self.hyper_params["epochs"] = epochs
        self.hyper_params["batch_size"] = loader["train"].batch_size
        self.hyper_params["train_ds_size"] = len_of_train_dataset

        fit_dict = {}
        # metrics = {}
        if validation:
            len_of_val_dataset = len(loader["val"].dataset)
            self.hyper_params["val_ds_size"] = len_of_val_dataset

        # self.experiment.log_parameters(self.hyper_params)
        acc_dic = {}
        loss_dic = {}
        acc_dict_val = {}
        loss_dict_val = {}
        for epoch in range(self._start_epoch, epochs):
            if checkpoint_path is not None and epoch % 100 == 0:
                self.save_to_file(checkpoint_path)
            # with self.experiment.train():
            correct = 0.0
            total = 0.0

            self.model.train()
            
            pbar = tqdm.tqdm(total=len_of_train_dataset)
            for x, y in loader["train"]:
                b_size = y.shape[0]
                total += y.shape[0]
                x = x.to(self.device) if isinstance(x, torch.Tensor) else [i.to(self.device) for i in x]
                y = y.to(self.device)

                pbar.set_description(
                    "\033[36m" + "Training" + "\033[0m" + " - Epochs: {:03d}/{:03d}".format(epoch+1, epochs)
                )
                pbar.update(b_size)

                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                # print("Outputs from fit: ",outputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y).sum().float().cpu().item()

                # self.experiment.log_metric("loss", loss.cpu().item(), step=epoch)
                # self.experiment.log_metric("accuracy", float(correct / total), step=epoch)
            acc_dic[epoch] = float(correct / total)
            loss_dic[epoch] = loss.cpu().item()
            if validation:
              # with self.experiment.validate():
                
                self.model.eval()
                with torch.no_grad():
                    val_correct = 0.0
                    val_total = 0.0
                    for x_val, y_val in loader["val"]:
                        val_total += y_val.shape[0]
                        x_val = x_val.to(self.device) if isinstance(x_val, torch.Tensor) else [i_val.to(self.device) for i_val in x_val]
                        y_val = y_val.to(self.device)

                        val_output = self.model(x_val)
                        val_loss = self.criterion(val_output, y_val)
                        _, val_pred = torch.max(val_output, 1)
                        val_correct += (val_pred == y_val).sum().float().cpu().item()
                        
                        # self.experiment.log_metric("loss", val_loss.cpu().item(), step=epoch)
                        # self.experiment.log_metric("accuracy", float(val_correct / val_total), step=epoch)
                    acc_dict_val[epoch] = float(val_correct / val_total)
                    loss_dict_val[epoch] = val_loss.cpu().item()

            pbar.close()
        fit_dict['train'] = [acc_dic, loss_dic]
        fit_dict['val'] = [acc_dict_val, loss_dict_val]
        print('train accuracy and loss per epoch: ', fit_dict['train'])
        print('validation accuracy and loss per epoch: ', fit_dict['val'])
        return fit_dict

    

    def evaluate(self, loader: DataLoader, verbose: bool = False, LOS = False) -> None or float:
        """
        The method of evaluating your PyTorch Model.
        With the assumption, This method use for training network for classification.

        ::

            clf = NeuralNetworkClassifier(
                    Network(), nn.CrossEntropyLoss(),
                    optim.Adam, optimizer_config, experiment
                )
            clf.evaluate(test_loader)


        :param loader: DataLoader for Evaluating: torch.utils.data.DataLoader
        :param verbose: bool
        :return: None
        """
        running_loss = 0.0
        running_corrects = 0.0
        pbar = tqdm.tqdm(total=len(loader.dataset))
        # for x,y in loader:
        #   print(x,y)
        metrics = {}
        self.model.eval()
        # self.experiment.log_parameter("test_ds_size", len(loader.dataset))
        # print('hello')
        # with self.experiment.test():
        with torch.no_grad():
          correct = 0.0
          total = 0.0
          auroc = []
          auprc = []
          minpse = []
          kappa = []
          mse = []
          mape = []
          for x, y in loader:
              b_size = y.shape[0]
              total += y.shape[0]
              x = x.to(self.device) if isinstance(x, torch.Tensor) else [i.to(self.device) for i in x]
              y = y.to(self.device)

              pbar.set_description("\033[32m"+"Evaluating"+"\033[0m")
              pbar.update(b_size)

              outputs = self.model(x)
              loss = self.criterion(outputs, y)
              # print("Outputs Printed: ",outputs)
              _, predicted = torch.max(outputs, 1)
              correct += (predicted == y).sum().float().cpu().item()

              running_loss += loss.cpu().item()
              running_corrects += torch.sum(predicted == y).float().cpu().item()
              if LOS:
                bin_met = print_metrics_regression(y, predicted)
                kappa.append(bin_met['kappa'])
                mse.append(bin_met['mse'])
                mape.append(bin_met['mape'])
              else:
                bin_met = print_metrics_binary(y, predicted)
                auroc.append(bin_met['auroc'])
                auprc.append(bin_met['auprc'])
                minpse.append(bin_met['minpse'])
          pbar.close()
        acc = float(running_corrects / total)
        if LOS:
          metrics['kappa'] = np.mean(kappa)
          metrics['mse'] = np.mean(mse)
          metrics['mape'] = np.mean(mape)
        else:
          metrics['auroc'] = np.mean(auroc)
          metrics['auprc'] = np.mean(auprc)
          metrics['minpse'] = np.mean(minpse)

      

        print("\033[33m" + "Evaluation finished. " + "\033[0m" + "Accuracy: {:.4f}".format(acc))
        print("All Metrics: ", metrics)

        
        return acc, metrics 

    def fit_phenotype(self, loader: DataLoader, epochs: int, checkpoint_path: str = None, validation: bool = True) -> None:
      len_of_train_dataset = len(loader['train'].dataset)
      epochs = epochs + self._start_epoch

      self.hyper_params["epochs"] = epochs
      self.hyper_params["batch_size"] = loader['train'].batch_size
      self.hyper_params["train_ds_size"] = len_of_train_dataset

      if validation:
          len_of_val_dataset = len(loader['val'].dataset)
          self.hyper_params["val_ds_size"] = len_of_val_dataset

      # self.experiment.log_parameters(self.hyper_params)
      acc_list = {}
      loss_list = {}
      acc_dict_val = {}
      loss_dict_val = {}
      for epoch in range(self._start_epoch, epochs):
          if checkpoint_path is not None and epoch % 100 == 0:
              self.save_to_file(checkpoint_path)
          # with self.experiment.train():
          correct = 0.0
          total = 0.0
          corrects = []
          losses = []
          self.model.train()
          pbar = tqdm.tqdm(total=len_of_train_dataset)
          for x, y in loader['train']:
              b_size = y.shape[0]
              total += y.shape[0]
              ## added this
              total += y.shape[1]
              x = x.to(self.device) if isinstance(x, torch.Tensor) else [i.to(self.device) for i in x]
              y = y.to(self.device)
              # print(y.shape)
              pbar.set_description(
                  "\033[36m" + "Training" + "\033[0m" + " - Epochs: {:03d}/{:03d}".format(epoch+1, epochs)
              )
              pbar.update(b_size)

              self.optimizer.zero_grad()
              outputs = self.model(x)
              # print(outputs)
              predicted = (outputs > 0.5).float()
              # predicted = torch.where(outputs >= 0.5, torch.tensor(1.0), torch.tensor(0.0))
              loss = self.criterion(outputs, y)
              
              # print(predicted)
              # print('y', y)
              loss.backward()
              self.optimizer.step()
              
              
              
              correct = torch.mean((predicted == y).float()).cpu().item()
              corrects.append(correct)
              losses.append(loss.cpu().item())
          acc_list[epoch] = float(np.mean(corrects))
          loss_list[epoch] = np.mean(losses)

          if validation:
              # with self.experiment.validate():
                
                self.model.eval()
                with torch.no_grad():
                    val_correct = 0.0
                    val_total = 0.0
                    val_corrects = []
                    val_losses = []

                    
                    for x_val, y_val in loader["val"]:
                        val_total += y_val.shape[0]
                        x_val = x_val.to(self.device) if isinstance(x_val, torch.Tensor) else [i_val.to(self.device) for i_val in x_val]
                        y_val = y_val.to(self.device)

                        val_output = self.model(x_val)
                        # val_predicted = torch.where(val_output >= 0.5, torch.tensor(1.0), torch.tensor(0.0))
                        val_loss = self.criterion(val_output, y_val)
                        val_predicted = (val_output > 0.5).float()
                        # _, val_pred = torch.max(val_output, 1)

                        # val_correct += (val_pred == y_val).sum().float().cpu().item()

                        # self.experiment.log_metric("loss", val_loss.cpu().item(), step=epoch)
                        # self.experiment.log_metric("accuracy", float(val_correct / val_total), step=epoch)
                        val_correct = torch.mean((val_predicted == y_val).float()).cpu().item()
                        val_corrects.append(val_correct)
                        val_losses.append(val_loss.cpu().item())
                    acc_dict_val[epoch] = float(np.mean(val_corrects))
                    loss_dict_val[epoch] = np.mean(val_losses)
                  # self.experiment.log_metric("loss", loss.cpu().item(), step=epoch)
                  # self.experiment.log_metric("accuracy", float(correct / total), step=epoch)
          pbar.close()
      # print(acc_list)
      # print(loss_list)
      # print(outputs)
      # print(predicted)
      fit_dict = {}
      fit_dict['train'] = [acc_list, loss_list]
      fit_dict['val'] = [acc_dict_val, loss_dict_val]
      print('train accuracy and loss per epoch: ', fit_dict['train'])
      print('validation accuracy and loss per epoch: ', fit_dict['val'])
      
      return fit_dict

    def evaluate_phenotype(self, loader: DataLoader, verbose: bool = False) -> None or float:
        running_loss = 0.0
        running_corrects = 0.0
        pbar = tqdm.tqdm(total=len(loader.dataset))

        self.model.eval()
        # self.experiment.log_parameter("test_ds_size", len(loader.dataset))
        acc_list = {}
        loss_list = {}
        metrics_all = {}
        # with self.experiment.test():
        with torch.no_grad():
            correct = 0.0
            total = 0.0
            corrects = []
            losses = []
            num_classes = None
            ave_auc_micro = []
            ave_auc_macro = []
            ave_auc_weighted = []

            for x, y in loader:
                b_size = y.shape[0]
                total += y.shape[0]
                x = x.to(self.device) if isinstance(x, torch.Tensor) else [i.to(self.device) for i in x]
                y = y.to(self.device)

                if num_classes is None:
                    num_classes = y.shape[1]

                pbar.set_description("\033[32m"+"Evaluating"+"\033[0m")
                pbar.update(b_size)

                outputs = self.model(x)
                ## this is wrong
                # outputs = torch.sigmoid(outputs)
                loss = self.criterion(outputs, y)
                # predicted = torch.where(outputs >= 0.5, torch.tensor(1.0), torch.tensor(0.0))
                predicted = (outputs > 0.5).float()
                running_corrects = torch.mean((predicted == y).float()).cpu().item()
                multi_pred = print_metrics_multilabel(y.reshape(-1),outputs.reshape(-1))
                ave_auc_micro.append(multi_pred['ave_auc_micro']) 
                ave_auc_macro.append(multi_pred['ave_auc_macro']) 
                ave_auc_weighted.append(multi_pred['ave_auc_weighted']) 


                running_loss = loss.cpu().item()
                corrects.append(running_corrects)
                losses.append(running_loss)
                

                # self.experiment.log_metric("loss", running_loss)

            # running_corrects /= num_classes
            # acc = float(torch.mean(running_corrects).cpu())
            # self.experiment.log_metric("accuracy", acc)
        metrics_all['ave_auc_micro'] = np.mean(ave_auc_micro)
        metrics_all['ave_auc_macro'] = np.mean(ave_auc_macro)
        metrics_all['ave_auc_weighted'] = np.mean(ave_auc_weighted)
        acc_list['overall_accuracy'] = float(np.mean(corrects))
        acc = float(np.mean(corrects))
        pbar.close()
        print("\033[33m" + "Evaluation finished. " + "\033[0m" + "Accuracy: {:.4f}".format(acc))
        print("All Metrics: ", metrics_all)
        
        return acc_list, metrics_all


    def save_checkpoint(self) -> dict:
        """
        The method of saving trained PyTorch model.

        Note,  return value contains
            - the number of last epoch as `epochs`
            - optimizer state as `optimizer_state_dict`
            - model state as `model_state_dict`

        ::

            clf = NeuralNetworkClassifier(
                    Network(), nn.CrossEntropyLoss(),
                    optim.Adam, optimizer_config, experiment
                )

            clf.fit(train_loader, epochs=10)
            checkpoints = clf.save_checkpoint()

        :return: dict {'epoch', 'optimizer_state_dict', 'model_state_dict'}
        """

        checkpoints = {
            "epoch": deepcopy(self.hyper_params["epochs"]),
            "optimizer_state_dict": deepcopy(self.optimizer.state_dict())
        }

        if self._is_parallel:
            checkpoints["model_state_dict"] = deepcopy(self.model.module.state_dict())
        else:
            checkpoints["model_state_dict"] = deepcopy(self.model.state_dict())

        return checkpoints

    def save_to_file(self, path: str) -> str:
        """
        | The method of saving trained PyTorch model to file.
        | Those weights are uploaded to comet.ml as backup.
        | check "Asserts".

        Note, .pth file contains
            - the number of last epoch as `epochs`
            - optimizer state as `optimizer_state_dict`
            - model state as `model_state_dict`

        ::

            clf = NeuralNetworkClassifier(
                    Network(), nn.CrossEntropyLoss(),
                    optim.Adam, optimizer_config, experiment
                )

            clf.fit(train_loader, epochs=10)
            filename = clf.save_to_file('path/to/save/dir/')

        :param path: path to saving directory. : string
        :return: path to file : string
        """
        if not os.path.isdir(path):
            os.mkdir(path)

        file_name = "model_params-epochs_{}-{}.pth".format(
            self.hyper_params["epochs"], time.ctime().replace(" ", "_")
        )
        path = path + file_name

        checkpoints = self.save_checkpoint()

        torch.save(checkpoints, path)
        # self.experiment.log_asset(path, file_name=file_name)

        return path

    def restore_checkpoint(self, checkpoints: dict) -> None:
        """
        The method of loading trained PyTorch model.

        :param checkpoints: dictionary which contains {'epoch', 'optimizer_state_dict', 'model_state_dict'}
        :return: None
        """
        self._start_epoch = checkpoints["epoch"]
        if not isinstance(self._start_epoch, int):
            raise TypeError

        if self._is_parallel:
            self.model.module.load_state_dict(checkpoints["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoints["model_state_dict"])

        self.optimizer.load_state_dict(checkpoints["optimizer_state_dict"])

    def restore_from_file(self, path: str, map_location: str = "cpu") -> None:
        """
        The method of loading trained PyTorch model from file.

        ::

            clf = NeuralNetworkClassifier(
                    Network(), nn.CrossEntropyLoss(),
                    optim.Adam, optimizer_config, experiment
                )
            clf.restore_from_file('path/to/trained/weights.pth')

        :param path: path to saved directory. : str
        :param map_location: default cpu: str
        :return: None
        """
        checkpoints = torch.load(path, map_location=map_location)
        self.restore_checkpoint(checkpoints)

    @property
    def experiment_tag(self) -> list:
        return self.experiment.get_tags()

    @experiment_tag.setter
    def experiment_tag(self, tag: str) -> None:
        """
        ::

            clf = NeuralNetworkClassifier(...)
            clf.experiment_tag = "tag"

        :param tag: str
        :return: None
        """
        if not isinstance(tag, str):
            raise TypeError

        self.experiment.add_tag(tag)

    @property
    def num_class(self) -> int or None:
        return self.__num_classes

    @num_class.setter
    def num_class(self, num_class: int) -> None:
        if not (isinstance(num_class, int) and num_class > 0):
            raise Exception("the number of class must be greater than 0.")

        self.__num_classes = num_class
        self.experiment.log_parameter("classes", self.__num_classes)

    def confusion_matrix(self, dataset: torch.utils.data.Dataset, labels=None, sample_weight=None) -> None:
        """
        | Generate confusion matrix.
        | result save on comet.ml.

        :param dataset: dataset for generating confusion matrix.
        :param labels: array, shape = [n_samples]
        :param sample_weight: array-lie of shape = [n_samples], optional
        :return: None
        """
        targets = []
        predicts = []
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        pbar = tqdm.tqdm(total=len(loader.dataset))

        self.model.eval()

        with torch.no_grad():
            for step, (x, y) in enumerate(loader):
                x = x.to(self.device)

                pbar.set_description("\033[31m" + "Calculating confusion matrix" + "\033[0m")
                pbar.update(step)

                outputs = self.model(x)
                _, predicted = torch.max(outputs, 1)

                predicts.append(predicted.cpu().numpy())
                targets.append(y.numpy())
            pbar.close()

        cm = pd.DataFrame(confusion_matrix(targets, predicts, labels, sample_weight))
        self.experiment.log_asset_data(
            cm.to_csv(), "ConfusionMatrix-epochs-{}-{}.csv".format(
                self.hyper_params["epochs"], time.ctime().replace(" ", "_")
            )
        )
