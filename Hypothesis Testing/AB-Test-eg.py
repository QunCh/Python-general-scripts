from scipy import stats
import pandas as pd
import os
import statsmodels.api as sm
import numpy as np

df = pd.read_csv('cookie_cats.csv')

### Summary Statistics
df.userid.nunique() == df.shape[0]
df.isnull().sum()
df['sum_gamerounds'].describe([0.01, 0.05, 0.1, 0.5, 0.8, 0.9, 0.95, 0.99])
df['sum_gamerounds'].plot.box(figsize=(5,10))

def q1(x):
    return x.quantile(0.25)
def q3(x):
    return x.quantile(0.75)
df.groupby(['version','retention_1']).agg({'sum_gamerounds': ['mean', 'median', 'count', 'max', 'std',q1,q3,pd.Series.nunique]})

### Remove Outlier
df = df[df.sum_gamerounds < df.sum_gamerounds.max()]

### 探索使用的样本数据
plot_df = df.groupby('sum_gamerounds').userid.count()
plot_df[plot_df.index>=30].head(20)
ax = plot_df[:100].plot(figsize = (10,6))
ax.set_title("The number of players that played 0-100 game rounds during the first week")
ax.set_ylabel("Number of Players")
ax.set_xlabel('# Game rounds')

### Feature Engineering
df["Retention"] = np.where((df.retention_1 == True) & (df.retention_7 == True), 1,0)
df.groupby(["version", "Retention"])["sum_gamerounds"].agg(["count", "median", "mean", "std", "max"])

df["NewRetention"] = list(map(lambda x,y: str(x)+"-"+str(y), df.retention_1, df.retention_7))
df.groupby(["version", "NewRetention"]).sum_gamerounds.agg(["count", "median", "mean", "std", "max"]).reset_index()


d30 = df[df.version == 'gate_30']
d40 = df[df.version == 'gate_40']

stats.levene(d30['sum_gamerounds'], d40['sum_gamerounds'])

### 平均值差值t检验
tt = stats.ttest_ind(d30['sum_gamerounds'], d40['sum_gamerounds'])
sm.stats.ttest_ind(d30['sum_gamerounds'], d40['sum_gamerounds'])
df.groupby(["version"])["sum_gamerounds"].agg(["count", "median", "mean", "std", "max"])

###手动计算
df.groupby('version').agg({'sum_gamerounds': [np.mean,np.std, np.size, np.var]})

### Proportions

### 1日留存
s1 = d30.retention_1.mean()*(1-d30.retention_1.mean())/d30.retention_1.count()
s2 = d40.retention_1.mean()*(1-d40.retention_1.mean())/d40.retention_1.count()
s = np.sqrt(s1+s2)

df.groupby('version').agg({'retention_1': [np.mean,np.std, np.size]})
df.groupby('version').agg([np.mean,np.std, np.size])

d = d30.retention_1.mean() - d40.retention_1.mean()
t = d/s
p = stats.t.sf(t, df=90187)*2
lcb = d-1.96*s
ucb = d+1.96*s
print('t: ', t, '\n','p: ',p,'\n' 'Confidence Interval: ', (lcb, ucb))
stats.t.ppf(q=1-.05/2,df=90187)

### 7日留存
s1 = d30.retention_7.mean()*(1-d30.retention_7.mean())/d30.retention_7.count()
s2 = d40.retention_7.mean()*(1-d40.retention_7.mean())/d40.retention_7.count()
s = np.sqrt(s1+s2)

df.groupby('version').agg({'retention_7': [np.mean,np.std, np.size,'var']})
#df.groupby('version').agg([np.mean,np.std, np.size])

d = d30.retention_7.mean() - d40.retention_7.mean()
t = d/s
p = stats.t.sf(t, df=90187)*2
lcb = d-1.96*s
ucb = d+1.96*s
print('t: ', t, '\n','p: ',p,'\n' 'Confidence Interval: ', (lcb, ucb))


tab1 = pd.pivot_table(df,index='version', columns='retention_1',values = 'userid', aggfunc='count')
tab2 = pd.pivot_table(df,index='version', columns='retention_7',values = 'userid', aggfunc='count')
t1 = pd.crosstab(df['version'], df['retention_1'])
t2 = pd.crosstab(df['retention_1'], df['version'])

stats.chi2_contingency(tab1)
stats.chi2_contingency(t2)
stats.chi2_contingency(tab2)
tab2

stats.norm.cdf(1.96)



# Define A/B groups
df["version"] = np.where(df.version == "gate_30", "A", "B")

# A/B Testing Function - Quick Solution
def AB_Test(dataframe, group, target):
    
    # Packages
    from scipy.stats import shapiro
    import scipy.stats as stats
    
    # Split A/B
    groupA = dataframe[dataframe[group] == "A"][target]
    groupB = dataframe[dataframe[group] == "B"][target]
    
    # Assumption: Normality
    ntA = shapiro(groupA)[1] < 0.05
    ntB = shapiro(groupB)[1] < 0.05
    # H0: Distribution is Normal! - False
    # H1: Distribution is not Normal! - True
    ###这里有问题，样本量足够大后，不需要检验是否正态分布吧
    if (ntA == False) & (ntB == False): # "H0: Normal Distribution"
        # Parametric Test
        # Assumption: Homogeneity of variances
        leveneTest = stats.levene(groupA, groupB)[1] < 0.05
        # H0: Homogeneity: False
        # H1: Heterogeneous: True
        
        if leveneTest == False:
            # Homogeneity
            ttest = stats.ttest_ind(groupA, groupB, equal_var=True)[1]
            # H0: M1 == M2 - False
            # H1: M1 != M2 - True
        else:
            # Heterogeneous
            ttest = stats.ttest_ind(groupA, groupB, equal_var=False)[1]
            # H0: M1 == M2 - False
            # H1: M1 != M2 - True
    else:
        # Non-Parametric Test
        ttest = stats.mannwhitneyu(groupA, groupB)[1] 
        # H0: M1 == M2 - False
        # H1: M1 != M2 - True
        
    # Result
    temp = pd.DataFrame({
        "AB Hypothesis":[ttest < 0.05], 
        "p-value":[ttest]
    })
    temp["Test Type"] = np.where((ntA == False) & (ntB == False), "Parametric", "Non-Parametric")
    temp["AB Hypothesis"] = np.where(temp["AB Hypothesis"] == False, "Fail to Reject H0", "Reject H0")
    temp["Comment"] = np.where(temp["AB Hypothesis"] == "Fail to Reject H0", "A/B groups are similar!", "A/B groups are not similar!")
    
    # Columns
    if (ntA == False) & (ntB == False):
        temp["Homogeneity"] = np.where(leveneTest == False, "Yes", "No")
        temp = temp[["Test Type", "Homogeneity","AB Hypothesis", "p-value", "Comment"]]
    else:
        temp = temp[["Test Type","AB Hypothesis", "p-value", "Comment"]]
    
    # Print Hypothesis
    print("# A/B Testing Hypothesis")
    print("H0: A == B")
    print("H1: A != B", "\n")
    
    return temp

# Apply A/B Testing
AB_Test(dataframe=df, group = "version", target = "sum_gamerounds")