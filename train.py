import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import shuffle
from network import BrainNet


def train_given_rule(X, y, meta_model, epochs, verbose = False, X_test = None, y_test = None):
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
        loss = meta_model(inputs, labels, epochs, batch, continue_ = continue_)

        all_rules.append(meta_model.output_rule.data.clone())

        if verbose and k % 5000 == 0: 
            print("Train on", k, " examples.")
            acc = evaluate(X, y, meta_model.network[0])
            train_accuracies.append(acc)
            print("Train Accuracy: {0:.4f}".format(acc))

            test_acc = evaluate(X_test, y_test, meta_model.network[0])
            test_accuracies.append(test_acc)
            print("Test Accuracy: {0:.4f}".format(test_acc))

    print("Final Acc.")
    print("Train on", k, " examples.")
    acc = evaluate(X, y, meta_model.network[0])
    train_accuracies.append(acc)
    print("Train Accuracy: {0:.4f}".format(acc))

    test_acc = evaluate(X_test, y_test, meta_model.network[0])
    test_accuracies.append(test_acc)
    print("Test Accuracy: {0:.4f}".format(test_acc))

    return train_accuracies, test_accuracies, all_rules


'''
    If fixed_rule == True, we only run GD on input layer and keep rule fixed
    Otherwise, apply GD to both input layer and local learning rule.
'''
def train_local_rule(X, y, meta_model, rule_epochs, epochs, batch, loss_ref, lr = 1e-2, fixed_rule = False, test_run = False, X_test = None, y_test = None, verbose = False):
    meta_model.double()
    if not test_run:
        optimizer = optim.Adam(meta_model.parameters(), lr=lr, weight_decay = 0.01)
    
    sz = len(X)

    print("INITIAL ACCURACY")
    acc = evaluate(X, y, meta_model.network[0])
    print("epoch 0","Accuracy: {0:.4f}".format(acc))

    running_loss = []
    all_rules = []
    print("Starting Train")
    for epoch in range(1, rule_epochs + 1):
        X, y = shuffle(X, y)

        print('Outer epoch ', epoch)

        cur_losses = []
        cur_accuracies = []
        test_accuracies = []
        for k in range(sz // batch):

            if not test_run:
                optimizer.zero_grad()

            inputs = X[k*batch:(k+1)*batch,:]
            labels = y[k*batch:(k+1)*batch]
            inputs = torch.from_numpy(inputs).double()
            labels = torch.from_numpy(labels).long()

            loss = meta_model(inputs, labels, epochs, batch)
            cur_losses.append(loss.item())
            if not test_run:
                loss.backward()
                optimizer.step()

            all_rules.append(meta_model.output_rule.data.clone())
            if verbose: 
                acc = evaluate(X, y, meta_model.network[0])
                cur_accuracies.append(acc)
                print("Train Accuracy: {0:.4f}".format(acc))
                if not (X_test is None):
                    test_acc = evaluate(X_test, y_test, meta_model.network[0])
                    test_accuracies.append(test_acc)
                    print("Test Accuracy: {0:.4f}".format(test_acc))
        
        loss = np.mean(cur_losses)
        loss_ref.append(loss.item())
        running_loss.append(loss.item())
        print(np.mean(running_loss[-10:]))

        if loss.item() < best_loss: 
            best_loss = loss.item() 
            best_output_rule = meta_model.output_rule 
            best_rule = meta_model.output_rule 

    if X_test is None: 
        return  running_loss, cur_accuracies, all_rules
    return running_loss, cur_accuracies, test_accuracies

def train_vanilla(X, y, model, epochs, batch, loss_ref, lr = 1e-2):
    X, y = shuffle(X, y)
    torch.set_printoptions(precision=3)

    model.double()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    sz = len(X)
    criterion = nn.CrossEntropyLoss()
    
    print("INITIAL ACCURACY")
    acc = evaluate(X, y, model)
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
        loss_ref.append(np.mean(cur_losses))
        if epoch % 1 == 0:
            print("Evaluating")
            acc = evaluate(X, y, model)
            print("epoch ", epoch, "Loss: {0:.4f}".format(running_loss[-1]), "Accuracy: {0:.4f}".format(acc))

    print('Finished Training')
    return running_loss


def evaluate(X, y, model):
    ac = [0] * model.m
    total = [0] * model.m
    with torch.no_grad():

        correct = 0

        outputs = model(torch.from_numpy(X).double())
        b = np.argmax(outputs, axis = 1).numpy()

        for i in range(len(b)):
            total[y[i]] += 1
            if b[i] == y[i]:
                ac[y[i]] += 1

        correct = np.sum(y == b)

        acc = correct / sum(total)

        for i in range(model.m):
            print("Acc of class", i, ":{0:.4f}".format(ac[i] / total[i]))
    return acc


