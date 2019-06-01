import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

df = pd.read_csv("/Users/dataset/ACOF_RCOF.csv",encoding='latin1')
#print(df.head())
#print(df.columns)

# Cleaning column labels
df = df.rename(index = str, columns = {"Slipped? (Y==1, N==0) SD=> 3 cm based on (HS-slipstop)": "slip_status"})

#df['slip_status']=df['slip_status'].apply(change_status)        
# Creating a new dataframe with slip status and ACOF-RCOF colmuns
cols_needed = [3, 7, 9, 11, 13, 15, 17, 19, 21]
df_acof_norm = df.iloc[:, cols_needed]
#df.columns = map(str.lower, df.columns) # Change the column names to lowercase
df_acof_norm.columns = df_acof_norm.columns.str.replace(",","_")
df_acof_norm.columns = df_acof_norm.columns.str.replace("ACOF","")
df_acof_norm.columns = df_acof_norm.columns.str.replace("-RCOFMean","")
print(df_acof_norm.columns)

np_acof_norm = df_acof_norm.values # df to array
np_acof_norm[:,0] = np_acof_norm[:,0].astype(int) # 'slip_status' chnaged to int
y = np_acof_norm[:,0]
model_logistic = LogisticRegression(C=1e5, solver='lbfgs')
beta0 = []
beta1 = []

for i in range(1,9):
    X = np_acof_norm[:,i]
    model_logistic.fit(X.reshape(-1,1),y)
    beta0.append(model_logistic.intercept_)
    beta1.append(model_logistic.coef_)

# Logistic curve
friction = np.arange(-0.2, 0.1, 0.001)
fig_LR, ax_LR = plt.subplots(figsize = (9,6))
ax_LR.set_ylim(0,1)
ax_LR.set_xlim(-0.2,0.1)
for j in range(0,8):
    exp_term = np.exp(beta0[j] + beta1[j] * friction)
    slip_risk = exp_term/(1+exp_term)
    ax_LR.plot(friction, slip_risk.reshape(-1,1), label = df_acof_norm.columns[j+1])
ax_LR.legend()
ax_LR.set_xlabel('ACOF-RCOF', fontsize='large', fontweight='bold')
ax_LR.set_ylabel('Probability of slip', fontsize='large', fontweight='bold')
ax_LR.set_title('Effect of biomechanical testing parameters on slip risk', fontsize='large', fontweight='bold')
ax_LR.spines['right'].set_visible(False) 
ax_LR.spines['top'].set_visible(False)
ax_LR.tick_params(axis="x", labelsize=12)
ax_LR.tick_params(axis="y", labelsize=12)
plt.savefig('/Users/Data science project/Slip_risk_models.png')
plt.show()

# AUC of ROC curve
AUC = []
for r in range(1,9):
    X = np_acof_norm[:,r]
    prob = model_logistic.predict_proba(X.reshape(-1,1)) # Probability estimates. 
    # prob: 1st col for no slip (0), 2nd col for slip (1)
    prob_of_slipping = prob[:, 1] # "The returned estimates for all classes are ordered by the label of classes."
    AUC_temp = roc_auc_score(y, prob_of_slipping)  
    AUC.append(AUC_temp)
   
# ROC curve for the lowest and highest AUC 
X_high = np_acof_norm[:,6] # '250N_ 17°_ 0.5m/s'
prob_high = model_logistic.predict_proba(X_high.reshape(-1,1))
prob_of_slipping_high = prob_high[:, 1] 
fpr_high, tpr_high, thresholds_high = roc_curve(y, prob_of_slipping_high)

X_low = np_acof_norm[:,1] # '250N_ 7°_0.3m/s'
prob_low = model_logistic.predict_proba(X_low.reshape(-1,1))
prob_of_slipping_low = prob_low[:, 1] 
fpr_low, tpr_low, thresholds_low = roc_curve(y, prob_of_slipping_low)

fig_roc, ax_roc = plt.subplots(figsize = (8,8))
ax_roc.plot(fpr_high, tpr_high, label = '250N_ 17°_ 0.5m/s, AUC = %0.3f' % AUC[5])
ax_roc.plot(fpr_low, tpr_low, label = '250N_ 7°_0.3m/s, AUC = %0.3f' % AUC[0])
ax_roc.plot([0, 1], [0, 1], color='black', lw= 2, linestyle='--')
ax_roc.legend()
ax_roc.set_ylabel('True positive rate (sensitivity)', fontsize='large', fontweight='bold')
ax_roc.set_xlabel('False positive rate (1-specificity)', fontsize='large', fontweight='bold')
ax_roc.set_xlim(-0.05,1.05)
ax_roc.set_ylim(-0.05,1.05)
ax_roc.tick_params(axis="x", labelsize=12)
ax_roc.tick_params(axis="y", labelsize=12)
ax_roc.tick_params(axis="x", labelsize=12)
ax_roc.tick_params(axis="y", labelsize=12)
ax_roc.set_title('ROC curves for the set of test parameters with highest and lowest AUC', fontsize='large', fontweight='bold')
plt.savefig('/Users/Data science project/ROC_curve.png')
plt.show()
