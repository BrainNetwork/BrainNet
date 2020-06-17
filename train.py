import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import shuffle
from network import BrainNet

np.set_printoptions(precision=4)

def train_given_rule(X, y, meta_model, verbose = False, X_test = None, y_test = None):
    all_rules = [] 
    test_accuracies = []
    train_accuracies = []

    batch = 1
    for k in range(len(X)): 
        inputs = X[k*batch:(k+1)*batch,:]
        labels = y[k*batch:(k+1)*batch]
        inputs = torch.from_numpy(inputs).double()
        labels = torch.from_numpy(labels).long()

        if k == 0: continue_ = False
        else: continue_ = True
        loss = meta_model(inputs, labels, 1, batch, continue_ = continue_)

        if k == len(X) - 1 or (verbose and k % 500 == 0): 
            print("Train on", k, " examples.")
            acc = evaluate(X, y, meta_model.m, meta_model.forward_pass)
            train_accuracies.append(acc)
            print("Train Accuracy: {0:.4f}".format(acc))

            test_acc = evaluate(X_test, y_test, meta_model.m, meta_model.forward_pass)
            test_accuracies.append(test_acc)
            print("Test Accuracy: {0:.4f}".format(test_acc))

    return train_accuracies, test_accuracies


'''
    If fixed_rule == True, we only run GD on input layer and keep rule fixed
    Otherwise, apply GD to both input layer and local learning rule.
'''
def train_local_rule(X, y, meta_model, rule_epochs, epochs, batch, lr = 1e-2, X_test = None, y_test = None, verbose = False):
    meta_model.double()

    optimizer = optim.Adam(meta_model.parameters(), lr=lr, weight_decay = 0.01)
    
    sz = len(X)

    running_loss = []
    print("Starting Train")
    for epoch in range(1, rule_epochs + 1):
        X, y = shuffle(X, y)

        print('Outer epoch ', epoch)

        cur_losses = []
        train_accuracies = []
        test_accuracies = []
        for k in range(sz // batch):

            optimizer.zero_grad()

            inputs = X[k*batch:(k+1)*batch,:]
            labels = y[k*batch:(k+1)*batch]
            inputs = torch.from_numpy(inputs).double()
            labels = torch.from_numpy(labels).long()

            loss = meta_model(inputs, labels, epochs, batch)
            cur_losses.append(loss.item())

            loss.backward()
            optimizer.step()

        if verbose or epoch == rule_epochs: 
            acc = evaluate(X, y, meta_model.m, meta_model.forward_pass)
            train_accuracies.append(acc)
            print("Train Accuracy: {0:.4f}".format(acc))
            if not (X_test is None):
                test_acc = evaluate(X_test, y_test, meta_model.m, meta_model.forward_pass)
                test_accuracies.append(test_acc)
                print("Test Accuracy: {0:.4f}".format(test_acc))
                
        loss = np.mean(cur_losses)
        running_loss.append(loss.item())
        print("LOSS:", np.mean(running_loss))

    return running_loss, train_accuracies, test_accuracies

def train_vanilla(X, y, model, epochs, batch, lr = 1e-2):
    X, y = shuffle(X, y)

    model.double()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    sz = len(X)
    criterion = nn.CrossEntropyLoss()
    
    print("INITIAL ACCURACY")
    acc = evaluate(X, y, model.m, model)
    print("epoch 0","Accuracy: {0:.4f}".format(acc))

    running_loss = []
    for epoch in range(1, epochs + 1):  
    
        cur_losses = []
        for k in range(sz//batch):

            inputs = X[k*batch:(k+1)*batch,:]
            labels = y[k*batch:(k+1)*batch]
            
            inputs = torch.from_numpy(inputs).double()
            labels = torch.from_numpy(labels).long()
            
            optimizer.zero_grad()

            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            cur_losses.append(loss.item())
    
        running_loss.append(np.mean(cur_losses))
        if epoch % 1 == 0:
            print("Evaluating")
            acc = evaluate(X, y, model.m, model)
            print("epoch ", epoch, "Loss: {0:.4f}".format(running_loss[-1]), "Accuracy: {0:.4f}".format(acc))

    print('Finished Training')
    return running_loss


def evaluate(X, y, num_labels, model_forward):
    ac = [0] * num_labels
    total = [0] * num_labels
    with torch.no_grad():

        correct = 0

        outputs = model_forward(torch.from_numpy(X).double())
        b = np.argmax(outputs, axis = 1).numpy()

        for i in range(len(b)):
            total[y[i]] += 1
            if b[i] == y[i]:
                ac[y[i]] += 1

        correct = np.sum(y == b)

        acc = correct / sum(total)

        for i in range(num_labels):
            print("Acc of class", i, ":{0:.4f}".format(ac[i] / total[i]))
    return acc


