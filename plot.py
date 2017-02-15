#!/usr/bin/python
#
# author:
# 
# date: 
# description:
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d

import sys

# Procedures
def spline_plot(x,y_train,y_eval,l,csv_file):
    plt.figure(figsize=(10, 8), dpi=100)
    
    #xvals = np.linspace(x.min(),x.max(),2000)
    #f1 = interp1d(x,y1,kind="cubic")
    #f2 = interp1d(x,y2,kind="cubic")
    #plt.plot(xvals, f1(xvals), label=l)
    #plt.plot(xvals, f2(xvals), label=l)
    
    mean_eval = y_eval.rolling(10).mean()
    std_eval  = y_eval.rolling(10).std()
    
    plt.plot(x,y_train, alpha=0.2)
    
    plt.plot(x,y_eval, color='g', alpha=0.2)
    plt.plot(x,mean_eval)
    plt.fill_between(x, mean_eval-std_eval, mean_eval+std_eval, color='b', alpha=0.1)
    
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=5, mode="expand", borderaxespad=0.)
    plt.savefig(csv_file + l + ".pdf", format="pdf")
    plt.show()

# Main

csv_file = sys.argv[1] if len(sys.argv) > 1 else ""

df = pd.read_csv(csv_file + ".stats.csv")

with plt.style.context('ggplot'):
    df_train = df[df["mode"]==0]
    df_eval = df[df["mode"]==1]
    
    spline_plot(df_train.epoch, df_train["len"]/20, df_eval["len"]/20, "length", csv_file)
    spline_plot(df_train.epoch, df_train["invalids"]/20, df_eval["invalids"]/20, "invalids", csv_file)
    spline_plot(df_train.epoch, df_train["quests_complete"], df_eval["quests_complete"], "quests", csv_file)
    spline_plot(df_train.epoch, df_train["deaths"], df_eval["deaths"], "deaths", csv_file)
    spline_plot(df_train.epoch, (df_train["score"]+2)/3, (df_eval["score"]+2)/3, "reward", csv_file)
    
    
    
    #plt.figure(figsize=(15, 6), dpi=80)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #       ncol=5, mode="expand", borderaxespad=0.)
    #plt.savefig(csv_file + ".train.pdf", format="pdf")
    #plt.show()

#with plt.style.context('ggplot'):
#    df_eval = df[df["mode"]==0]
#    plt.figure(figsize=(20, 6), dpi=80)
#    spline_plot(df_eval.epoch, df_eval["len"]/20, "length")
#    spline_plot(df_eval.epoch, df_eval["invalids"]/20, "invalids")
#    spline_plot(df_eval.epoch, df_eval["quests_complete"], "quests")
#    spline_plot(df_eval.epoch, df_eval["deaths"], "deaths")
#    spline_plot(df_eval.epoch, (df_eval["score"]+2)/3, "reward")
#    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#           ncol=5, mode="expand", borderaxespad=0.)
#    plt.savefig(csv_file + ".eval.pdf", format="pdf")
#    plt.show()

def stats(csv_file, mode):
    df = pd.read_csv("{}.{}.csv".format(csv_file,mode), delimiter=",", quotechar="'")

    df_qc = df[df.reward==1.0]
    df_qn = df[df.reward!=1.0]

    print "qc", len(df_qc.index)

    print "qc sleepy", df_qc.sleepy.sum()
    print "qc bored", df_qc.bored.sum()
    print "qc hungry", df_qc.hungry.sum()

    print "qc sleepy quest",        len(df_qc[(df_qc.sleepy!=1) & (df_qc.quest==0)].index)
    print "qc sleepy quest + hint", len(df_qc[(df_qc.sleepy==1) & (df_qc.quest==0)].index)
        
    print "qc bored quest",        len(df_qc[(df_qc.bored !=1) & (df_qc.quest==1)].index)
    print "qc bored quest + hint", len(df_qc[(df_qc.bored ==1) & (df_qc.quest==1)].index)
    
    print "qc hungry quest",        len(df_qc[(df_qc.hungry !=1) & (df_qc.quest==2)].index)
    print "qc hungry quest + hint", len(df_qc[(df_qc.hungry ==1) & (df_qc.quest==2)].index)
    
    print ""
    
    print "qn", len(df_qn.index)
    
    print "qn sleepy", df_qn.sleepy.sum()
    print "qn bored", df_qn.bored.sum()
    print "qn hungry", df_qn.hungry.sum()

    print "qn sleepy quest + hint", len(df_qn[(df_qn.sleepy==1) & (df_qn.quest==0)].index)
    print "qn bored quest + hint", len(df_qn[(df_qn.bored ==1) & (df_qn.quest==1)].index)
    print "qn hungry quest + hint", len(df_qn[(df_qn.hungry==1) & (df_qn.quest==2)].index)

stats(csv_file,0)
print "\n>\n"
stats(csv_file,1)

#plt.plot(df_train.epoch, df_train.invalids, 'o')

