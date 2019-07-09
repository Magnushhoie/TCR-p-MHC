#imports
print("Importing ...")
import optparse, os, sys, glob, re, pickle, time, datetime
sys.path.insert(0, "/home/maghoi/Magnus_pMHC/")
import numpy as np
import pandas as pd

#FastAI functions
from fastai.basic_data import *
from fastai.basic_train import *
from fastai.callbacks import *
from fastai.data_block import *
from fastai.metrics import *
from fastai.train import *
from fastai.utils import *
from fastai.core import *
from fastai.gen_doc import *

#Disable training progress bar (set next to learner fit)
import fastprogress
fastprogress.fastprogress.NO_BAR = True
master_bar, progress_bar = fastprogress.force_console_behavior()
master_bar, progress_bar = master_bar, progress_bar


#Pytorch
import torch
import torch.nn as nn
import torch.utils.data as tdatautils

#Stats
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp

from sklearn import metrics, svm, datasets, random_projection
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc, roc_auc_score, matthews_corrcoef, average_precision_score 
from sklearn.preprocessing import Normalizer, MinMaxScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

#Dataloaders and custom scripts
from allscripts import data_generator, data_generator_blosum, data_generator_filenames
from allscripts import upsample, generate_weights

#Set random seeds
torch.cuda.manual_seed_all(1)
np.random.seed(1)

#############################
# Parse commandline options
#############################

parser = optparse.OptionParser()

#Set architecture
parser.add_option("--modules", dest="MODULES", default=5, help="Set number of convolutional modules in network")
parser.add_option("--residuals", dest="RESIDUALS", default=3, help="Set number of residuals. 3 = added to 3rd module")
parser.add_option("--onecycle", dest="ONECYCLE", default=True, help="Use 1-cycle policy or fit normally")
parser.add_option("--rp", dest="RP", default=False, help="Whether to use random projection unit")

#Set mode, and random or ordered partitions
parser.add_option("-m", "--mode", dest="MODE", default=5, help="Set training mode: 1. Cross-val, 2. Nested-cross-val")
parser.add_option("--r", "--random", dest="RANDOM", default=False, help="Set (random) partitions from filenames with True. Default ordered partitions")
parser.add_option("--sets", dest="SETS", default=4, help="Number of times to train the network (e.g. 2 sets of 4 cycles)")

#Set outdir, comment and csvfile path
parser.add_option("-o", "--outdir", dest="OUTDIR", default="/scratch/maghoi/data/1may/", help="Set number of features used for each sequence position in X input data")
parser.add_option("-c", dest="COMMENT", default="", help="Commen for CSV file")
parser.add_option("--csvfile", dest="CSVFILE", default="/home/maghoi/Magnus_pMHC/CSV/CSV_FF.csv")

#Network parameters (0:21, aminos, 21:59) (0:59 features are structural, 59: features energy)
parser.add_option("--x1", "--x1", dest="X1", default=0, help="Set starting position for features used for each sequence position in X input data")
parser.add_option("--x2", "--x2", dest="X2", default=53, help="Set number of features used for each sequence position in X input data")
parser.add_option("--x3", "--x3", dest="X3", default=0, help="Set starting position for features used for each sequence position in X input data")
parser.add_option("--x4", "--x4", dest="X4", default=0, help="Set number of features used for each sequence position in X input data")
parser.add_option("--x5", "--x5", dest="X5", default=0, help="Set number of features used for each sequence position in X input data")
parser.add_option("--x6", "--x6", dest="X6", default=0, help="Set number of features used for each sequence position in X input data")
parser.add_option("--dp", dest="DP", default=0.2, help="Drop-prob")
parser.add_option("--lr", dest="LR", default=1, help="Factor to divide LR by")

#Set masking for amino acids within any 2 regions
# E.g. mask peptide sequence from 181-192 (--m1 181 --m2 192)
parser.add_option("--m1", "--mask1", dest="MASK1", default="", help="Set masking for any region in X")
parser.add_option("--m2", "--mask2", dest="MASK2", default="", help="Set masking for any region in X")
parser.add_option("--m3", "--mask3", dest="MASK3", default="", help="Set masking for any region in X")
parser.add_option("--m4", "--mask4", dest="MASK4", default="", help="Set masking for any region in X")



# Print network (layer) sizes
parser.add_option("-p", "--ps", dest="ps", default=False, help="Print network sizes")

#Remove later
# Load baseline?
parser.add_option("-l", "--load", dest="LOAD", default="", help="Load baseline trained models")

(options,args) = parser.parse_args()

MODE = int(options.MODE)
OUTDIR = str(options.OUTDIR)
COMMENT = str(options.COMMENT)
CSVFILE = str(options.CSVFILE)
PS = options.ps
SETS = int(options.SETS)
X1 = int(options.X1)
X2 = int(options.X2)
X3 = int(options.X3)
X4 = int(options.X4)
X5 = int(options.X5)
X6 = int(options.X6)

print("Input parameters:")
if options.MODULES != 5:
    COMMENT += "Mod:"+str(options.MODULES)
    print("Modules:", str(options.MODULES))

if options.RESIDUALS != 4:
    COMMENT += "Res:"+str(options.RESIDUALS)
    print("Residuals:", str(options.RESIDUALS))
    RESIDUALS = int(int(options.RESIDUALS) - 1)
else:
    RESIDUALS = int(int(options.RESIDUALS) - 1)

if options.ONECYCLE != True:
    COMMENT += " no1c"
    print("Using standard train, not using 1cycle:")

if options.RANDOM != False:
    RANDOM = True
    COMMENT += " RP"
    print("Random partition mode set")
    
if options.RP != False:
    RP = True
    COMMENT += " unitrp"
    print("Random project unit on")
else:
    RP = False

if options.SETS != int(1):
    COMMENT += " S:" + str(SETS)
    print("Sets:", str(SETS))
    
if options.LR != int(1):
    COMMENT += " LR:" + str(options.LR)
    print("LR divided by:", str(options.LR))

if options.X1 != int(0) or options.X2 != int(53):
    COMMENT += " X:" + str(X1) + "_" + str(X2)
    print("X1:", str(X1) + " to " + str(X2))
if options.X3 != int(0) or options.X4 != int(0):
    COMMENT += " X3:" + str(X3) + "_" + str(X4)
    print("X3:", str(X3) + " to " + str(X4))
if options.X5 != int(0) or options.X6 != int(0):
    COMMENT += " X5:" + str(X5) + "_" + str(X6)
    print("X5:", str(X5) + " to " + str(X6))

if options.MASK1 != "" and options.MASK2 != "":
    MASK1 = int(options.MASK1)
    MASK2 = int(options.MASK2)
    COMMENT += " M:" + str(MASK1) + "_" + str(MASK2)
    print("Masking in region", str(MASK1) + str(MASK2))
    
if options.MASK3 != "" and options.MASK4 != "":
    MASK3 = int(options.MASK3)
    MASK4 = int(options.MASK4)
    COMMENT += " M2:" + str(MASK3) + "_" + str(MASK4)
    print("Masking in region", str(MASK3) + str(MASK4))
    
if options.DP != 0.2:
    COMMENT += " dp:" + str(options.DP)
    print("Drop prob set to", str(options.DP))

print("Comment:", COMMENT)
print("Model outdir:", OUTDIR)
print("CSV path:", CSVFILE)

os.makedirs(OUTDIR, exist_ok = True)

#Remove later
baseline = ["/scratch/maghoi/data/7may/baseline/TXV0_1234",
    "/scratch/maghoi/data/7may/baseline/TXV1_0234",
    "/scratch/maghoi/data/7may/baseline/TXV2_0134",
    "/scratch/maghoi/data/7may/baseline/TXV3_0124",
    "/scratch/maghoi/data/7may/baseline/TXV4_0123"]

#############################
# Functions
#############################

#Convert numpy to torch
def to_torch_data(x,np_type,tch_type):
    return torch.from_numpy(x.astype(np_type)).to(tch_type)

#Print tail of CSV file
def csvfile(n = 10, filename = CSVFILE):
    df = pd.read_csv(filename)
    return df.tail(n)
    
#Main function for calculating model performance
def record_stats(ds = DatasetType.Valid):
    #Get raw predictions
    preds = learn.get_preds(ds)
    outputs = preds[0]
    targets = preds[1]

    #Find highest multi-class prediction (yes, this is wrong ...)
    yhat = []

    for i in range(len(outputs)):
        pred = outputs[i].tolist()
        pred = pred.index(max(pred))
        yhat.append(pred)

    #Pairwise comparison
    yhat = np.array(yhat)
    y_true = np.array(targets)
    y_scores = outputs[:, 1]
    y_scores_binary = np.where(y_scores > 0.5, 1, 0)

    correct = yhat == y_true
    auc = roc_auc_score(y_true, y_scores)
    mcc = matthews_corrcoef(y_true, y_scores_binary)
    f1 = f1_score(y_true, y_scores_binary, average="binary")
    avp = average_precision_score(y_true, y_scores)

    correct = round(sum(correct) / len(targets), 3)
    auc = round(auc, 3)
    mcc = round(mcc, 3)
    f1 = round(f1, 3)
    avp = round(avp, 3)

    confusion = metrics.confusion_matrix(y_true, yhat)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, yhat).ravel()
    tpr = (tp / (tp+fn))
    tnr = (tn / (tn+fp))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    tpr = round(tpr, 3)
    tnr = round(tnr, 3)
    precision = round(precision, 3)
    recall = round(recall, 3)
    
    return(correct, auc, mcc, f1, avp, tpr, tnr, precision, recall, confusion, y_scores, y_scores_binary, y_true)

# Save performance sets, hyperparam and comment to CSVfile
now = time.time()
def stats_to_csv(start_time = time.time(), val_part = 4, test_part = int(), comment = "", LR="", ds=DatasetType.Valid):
    train_str = [0, 1, 2, 3, 4]
    val_str = val_part
    test_str = test_part
    
    stat_df = pd.DataFrame(columns = ["Comment", "Test", "Validation", "Training", "Correct", "AUC", "MCC", "F1", "AVP", "TPR", "TNR", "Prec", "Rec", "Confusion matrix", "LR", "Duration (s)", "Timestamp", "y_hat", "y_hat_binary", "y_true"])
    
    #Check for CSV file
    if not glob.glob(CSVFILE):
        stat_df.to_csv(CSVFILE, mode = "w", header = True, index = False)

    #Remove val / test from training parts
    train_str.remove(val_str)
    if test_part != int():
        train_str.remove(test_str)
    
    #Get model performance
    data = record_stats(ds = ds)
    
    #Extract and remove preds from data
    y_hat = list(np.array(data[-3]))
    y_hat_binary = list(np.array(data[-2]))
    y_true = list(np.array(data[-1]))
    data = data[:-3]
    
    duration = round(time.time() - start_time)
    timestamp = str(datetime.datetime.now())
    
    #Add to stat_df and save to CSV
    row = [comment, test_str, val_str, train_str] + list(data) + [LR, duration, timestamp, y_hat, y_hat_binary, y_true]
    stat_df.loc[len(stat_df)] = row
    stat_df.to_csv(CSVFILE, mode = "a", header = False, index = True)
    print(["ACC", "AUC", "MCC", "F1", "AVP", "TPR", "TNR", "Prec", "Rec", "Confusion matrix"])
    print(row[4:13])
    print(row[13])
    #return(pd.DataFrame(row[4:13]).transpose())


#############################
# Load data
#############################

if options.RANDOM == "True":
    print("Loading data by filename (randomized) partitions")
    filelist = glob.glob("/home/maghoi/pMHC_data/features10/*")
    p0 = filelist[0:293]
    p1 = filelist[293:586]
    p2 = filelist[586:879]
    p3 = filelist[879:1172]
    p4 = filelist[1172:1464]
    
else:
    print("Loading data by ordered partitions ...")
    p0 = glob.glob("/home/maghoi/pMHC_data/features12/*0p*")
    p1 = glob.glob("/home/maghoi/pMHC_data/features12/*1p*")
    p2 = glob.glob("/home/maghoi/pMHC_data/features12/*2p*")
    p3 = glob.glob("/home/maghoi/pMHC_data/features12/*3p*")
    p4 = glob.glob("/home/maghoi/pMHC_data/features12/*4p*")

#train = p0 + p1 + p2
#valid = p3
#test = p4

print("Partition sizes:", len(p0), len(p1), len(p2), len(p3), len(p4))


#############################
# Model Start
#############################

print("Setting up model ...")

#Setting number of features
x_one = list(range(X1, X2))
x_two = list(range(X3, X4))
x_three = list(range(X5, X6))
features = x_one + x_two + x_three

    
#Hyperparams for CNN

criterion = nn.CrossEntropyLoss()
in_channel = 468*3
n_hid = 1
epochs = 20
batch_size = 32

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        #Batchnorm0
        self.a_norm_0 = nn.BatchNorm1d(in_channel)
        
        #Prediction module
        self.a_Linear = nn.Linear(in_features = in_channel, out_features = 2)
        #self.a_BatchNorm = nn.BatchNorm1d(f*n_hid)
        #self.a_ReLU = nn.ReLU(f*n_hid)
        #self.a_Linear2 = nn.Linear(in_features = f*n_hid, out_features = 2)

    def forward(self, x):
        global ps
        bs0 = x.shape[0]
        x = x[:, :, 0:3]
        if ps: print("\nInput", x.shape)
            
        #x = x.transpose(1, 2)
        #if ps: print("Transpose", x.shape)
            
        #x = x.view(bs0, (468*21))
        x = torch.reshape(x, (bs0, 468*3))
        if ps: print("View", x.shape)
            
        x = self.a_norm_0(x)
        if ps: print("Norm", x.shape)
                    
        #Prediction module
        allparts = self.a_Linear(x)
        if ps: print("Linear", allparts.shape)
        
        #allparts = allparts.view(bs0, 2)
        #if ps: print("Below", allparts.shape)
        ps = False
        
        x = allparts
        return x

ps = True
    
#############################
# Run model using early-stopping
#############################

if MODE == 5:
    print("Mode: 5. Early-stopping")
    
def early_stop2(cycles=20, LR=16*[1e-02, 1e-01, 5e-03, 5e-02], epochs=16*[1],
                         val_part = int(4), test_part = int(),
                         comment="", outdir = "/scratch/maghoi/data/models/"):
    train_part=[0, 1, 2, 3, 4]
    #LR multiplied by LR factor. Default 1
    LR = pd.Series(LR)/int(options.LR)
    now = time.time()
    stat_df = pd.DataFrame(columns=["Correct", "AUC", "MCC", "F1", "AVP", "TPR", "TNR", "Prec", "Rec", "Confusion"])
    
    #Create outidr
    os.makedirs(outdir, exist_ok = True)
    os.makedirs(outdir+"/saved/", exist_ok = True)
    
    no_improv = 0
    best = 0
    for i in range(0, cycles):  
        print("Cycle:", i+1, "/", cycles, "LR:", LR[i*2], "->", LR[(i*2)+1])
        
        #Train model
        now = time.time()
        
        if options.ONECYCLE == True:
            learn.fit_one_cycle(epochs[i], max_lr=slice(None, LR[i*2], LR[(i*2)+1]), wd = 0.01)
        else:
            print("Not using 1-cycle policy")
            learn.fit(epochs[i], lr=slice(None, LR[i*2], None), wd = 0.01)

        if i == 0:
            print("Saving initial model ...", outdir+"temp_model")
            learn.save(outdir + "temp_model")
        
        if i >= 1:
            #Check performance best vs now
            stats = pd.DataFrame(record_stats()[0:10]).transpose()
            stats.columns = ["Correct", "AUC", "MCC", "F1", "AVP", "TPR", "TNR", "Prec", "Rec", "Confusion"]
            stat_df = stat_df.append(stats)
            
            #df = pd.read_csv("/home/maghoi/main/data/Stats1.csv")
            #before = float(stat_df.iloc[len(stat_df)-(2)]["MCC"])
            now = float(stat_df.iloc[len(stat_df)-(1)]["MCC"])
            print("Best;", best, "vs", "now:", now)

            #Load model before if performance worse
            if now > best:
                print("MCC higher, saving ...")
                learn.save(outdir + "temp_model")
                no_improv = 0
                best = float(now)

                
    #Save model
    train_part.remove(val_part)
    if test_part == int():
        test_part = "0"
    else:
        train_part.remove(test_part)

    test_str = str(test_part)
    val_str = str(val_part)
    train_str = "".join(map(str, train_part))

    filepath = outdir+"saved/" + "T" + test_str + "V" + val_str + "_" + train_str
    learn.load(outdir+"temp_model")
    learn.save(filepath)
    
    filelist = glob.glob(outdir + "temp_model*")
    if filelist:
        print("Removing", filelist[0])
        os.remove(filelist[0])
    else:
        print("No file found????")
    
    return(filepath)
    print("Done")


### Early stop model
if MODE == 5:
    print("Mode: 5. Early-stopping, BLOSUM data")
    
    print("Loading data ...")
    X_mhc = pd.read_pickle("/home/maghoi/pMHC_data/02_Features/X_mhc.pickle")
    X_pep = pd.read_pickle("/home/maghoi/pMHC_data/02_Features/X_pep.pickle")
    X_tcr_a = pd.read_pickle("/home/maghoi/pMHC_data/02_Features/X_tcr_a.pickle")
    X_tcr_b = pd.read_pickle("/home/maghoi/pMHC_data/02_Features/X_tcr_b.pickle")
    PAR_VEC = pd.read_pickle("/home/maghoi/pMHC_data/02_Features/PAR_VEC.pickle")
    target_y = pd.read_pickle("/home/maghoi/pMHC_data/02_Features/target_y.pickle")
    names = pd.read_pickle("/home/maghoi/pMHC_data/02_Features/names.pickle")
    
    def to_torch_data(x,np_type,tch_type):
        return torch.from_numpy(x.astype(np_type)).to(tch_type)

    df = pd.DataFrame(index = range(0,7))
    #simple learn on BLOSUM
    for i in range(0, 5):
        #Load
        X, y, X_val, y_val, X_test, y_test = data_generator_blosum(X_mhc, X_pep, X_tcr_a, X_tcr_b, target_y,
                                                                   names, PAR_VEC, norm = True, val_part = i)

        X0, y0, X0_val, y0_val, X0_test, y0_test = X.copy(), y.copy(), X_val.copy(), y_val.copy(), X_test.copy(), y_test.copy()

        #Upsample
        Xp, yp = upsample(X, y)
        Xp_val, yp_val = upsample(X_val, y_val)
        Xp_test, yp_test = upsample(X_test, y_test)
        
        #Map to train data and send to cuda
        X, X_val, X_test = map(lambda x: to_torch_data(x,float,torch.float32),(Xp, Xp_val, Xp_test))
        y, y_val, y_test = map(lambda x: to_torch_data(x,int,torch.int64),(yp, yp_val, yp_test))

        train_ds = tdatautils.TensorDataset(X, y)
        valid_ds = tdatautils.TensorDataset(X_val, y_val)
        test_ds = tdatautils.TensorDataset(X_test, y_test)

        X0, X0_val, X0_test = map(lambda x: to_torch_data(x,float,torch.float32),(X0, X0_val, X0_test))
        y0, y0_val, y0_test = map(lambda x: to_torch_data(x,int,torch.int64),(y0, y0_val, y0_test))

        train_ds0 = tdatautils.TensorDataset(X0, y0)
        valid_ds0 = tdatautils.TensorDataset(X0_val, y0_val)
        test_ds0 = tdatautils.TensorDataset(X0_test, y0_test)


        batch_size = 32
        ps = False
        my_data_bunch = DataBunch.create(train_ds, valid_ds0, test_ds0, bs=batch_size)
        net = Model().cuda()

        learn = Learner(my_data_bunch,
                             net,
                             opt_func=torch.optim.Adam,
                             loss_func=criterion, metrics=[accuracy],
                             callback_fns=[partial(EarlyStoppingCallback, min_delta=0.01, patience=3)],
                             wd = 0.01)

        SETS = 20
        cycles = SETS
        epochs = 2*[1]+(SETS-2)*[1]
        #LR = [1e-02, 1e-01, 1e-02, 1e-01, 1e-02, 1e-01]+(SETS-3)*[5e-03, 5e-02]

        LR = [5e-03, 5e-02]*20

        #filepath = simple_train(cycles = SETS*4, epochs = SETS*4*[1], LR = [1e-02, 1e-01, 1e-02, 1e-01, 1e-03, 5e-02, 1e-03, 5e-02], val_part = i, skip = False)
        filepath = early_stop2(cycles = cycles, epochs = epochs, LR = LR, val_part = i)

        stats_to_csv(val_part = i, comment = "blosum")
        
        print("Model saved:\n", filepath)
        learn.load(filepath)
        stats_to_csv(comment = COMMENT, val_part = i, LR=LR, ds = DatasetType.Valid)

        
#############################
# Model train function
#############################
# Cross validation with specified LR, epochs per cycle and number of cycles
# Saves model

def simple_train(cycles=4, LR=10*[1e-02, 1e-01], epochs=10*[3],
                         val_part = int(4), test_part = str(),
                         skip = False):
    train_part=[0, 1, 2, 3, 4]
    LR = pd.Series(LR)
    
    #Skip validation
    if skip == True:
        learn.data.valid_dl = None
    
    #Train model
    for i in range(0, cycles):  
        print("Cycle:", i+1, "/", cycles, "Epochs:", epochs[i], "LR:", LR[i*2], "->", LR[(i*2)+1])
        now = time.time()
        learn.fit_one_cycle(epochs[i], max_lr=slice(None, LR[i*2], LR[(i*2)+1]), wd = 0.01)
    
    #Save model to filepath
    train_part.remove(val_part)
    if test_part == str():
        test_part = "X"
    else:
        train_part.remove(test_part)

    test_str = str(test_part)
    val_str = str(val_part)
    train_str = "".join(map(str, train_part))

    filepath = OUTDIR + "T" + test_str + "V" + val_str + "_" + train_str
    learn.save(filepath)
    
    return(filepath)

