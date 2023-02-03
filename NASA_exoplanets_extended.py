# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 13:43:09 2022

@author: gabor
"""

## Goal if this code is to practice data extraction and analysis using the NASA
## exoplanets database. The base of this code is a tutorial done by Anton Petrov:
## https://www.youtube.com/watch?v=Lu31biXStWQ.

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder

np.set_printoptions(precision=2)
sns.set({'figure.figsize':(13,13)}, style="ticks", )


ref_csv = *** INSERT PATH HERE ***
planets = pd.read_csv(ref_csv, sep=",", comment="#")


# inactive correlation plot: mass - discovery method
#
#sns.boxplot(x="pl_bmassj", y="discoverymethod", data=planets)
#sns.swarmplot(x="pl_bmassj", y="discoverymethod", data=planets, size=1, color=".3", linewidth=0)


##
## DATA EXTRACTION
##

df = pd.DataFrame(planets)
cols = [0,3,4,6,22,24,26,29,30,32,33,34,38,39,40,44,50] # preselected parameters after checking the raw data
df = df[df.columns[cols]]

df = df.dropna(how='any')           # assign back
df.dropna(how='any', inplace=True)  # set inplace parameter
df.rename(columns = {"pl_name":"Name",
                     "sy_snum": "Star #",
                     "sy_pnum": "Planet #",
                     "discoverymethod": "Disc. Method",
                     "pl_orbper": "Orb. Period [days]",
                     "pl_rade": "Pl. Radius",
                     "pl_bmasse": "Mass [Earth]",
                     "pl_dens": "Pl. Density",
                     "pl_orbeccen": "Eccentricity",
                     "pl_eqt": "Equilibrium Temp [K]",
                     "ttv_flag": "Transit Timing Variations",
                     "pl_trandep": "Transit Depth",
                     "st_teff": "Sol. eff. T",
                     "st_rad": "Stel. rad.",
                     "st_mass": "Stel. mass",
                     "st_dens": "Stel. dens.",
                     "sy_dist": "Distance"}, inplace = True)

#df.drop("Planet Name")
#print(df)

# number_of_entries=df.shape[0]
# planet_number_limit=number_of_entries     #limiting number of planets for ease of calculation
# df=df.drop( df.index.to_list()[planet_number_limit:] ,axis = 0 )

##
##  Correlation calculation and heatmap visualization
##

print('\n\n Correlation analysis with heatmap \n\n')

f, ax = plt.subplots(figsize=(15,15))

corr_matrix=df.corr()

cuns=corr_matrix.unstack()   # pivot
corrlist=cuns.sort_values(ascending=False)            # sorting and filtering the
corrlist=corrlist.drop_duplicates(keep="first")       # top 10 correlation pairs
corrlist_indexes=corrlist.index.values.tolist()[1:10] # for the heatmap
#corrlist.columns= ["Parameter 1","Parameter 2", "Correlation"]

print("Parameters with highest correlation: ")
print(corrlist[1:10])
#print(corrlist[0])

sns.heatmap(corr_matrix, annot=True, square=True)
ax.set_title("Exoplanet parameter correlation matrix\n", {'fontsize':25})
plt.show()

ind1=corrlist_indexes[0][0]
ind2=corrlist_indexes[0][1]
#print("First index:",ind1,"Second index:",ind2)

keep_correlated_columns= [ind1,ind2]
df_corr=df.reindex(columns=keep_correlated_columns)
#df_corr=df_corr.drop(axis=0)


##
## DATA ANALYSIS
##

print("\n\n Data analysis \n\n")
#pd.set_eng_float_format(accuracy=2, use_eng_prefix=True)

#df2=df.mean(axis='index')
df2=df.agg({'Star #' : ['min', 'max','mean', 'sum'], 'Planet #' : ['min', 'max','mean', 'sum'], 'Orb. Period [days]' : ['min', 'max','mean', 'sum']}, axis='index')
print(df2)
print()
df2=df.agg({"Pl. Radius" : ['min', 'max','mean', 'sum'], "Mass [Earth]" : ['min', 'max','mean', 'sum'], "Pl. Density" : ['min', 'max','mean', 'sum']}, axis='index')
print(df2)

av_stars=df["Star #"].mean()
av_orbper=df["Orb. Period [days]"].mean()
av_plmass=df["Mass [Earth]"].mean()

df_average=df
df_average=df_average.drop( df_average.index.to_list()[1:] ,axis = 0 )

df_average["Planet Name"]="Average"
df_average["Star #"]=av_stars
df_average["Orb. Period [days]"]=av_orbper
df_average["Mass [Earth]"]=av_plmass


star1=np.sum(df["Star #"] == 1)
star2=np.sum(df["Star #"] == 2)
star3=np.sum(df["Star #"] == 3)
star4=np.sum(df["Star #"] == 4)
starsum=star1+star2+star3+star4


print("\nPlanets around a single star:",star1,", that is", np.round(star1/starsum,5)*100,"%")
print("Planets around two stars:",star2,", that is", np.round(star2/starsum,5)*100,"%")
print("Planets around three stars:",star3,", that is", np.round(star3/starsum,5)*100,"%")
print("Planets around four stars:",star4,", that is", np.round(star4/starsum,5)*100,"%")

below_earthmass=np.sum(df["Mass [Earth]"]<=1)
over_earthmass=np.sum(df["Mass [Earth]"]>1)

print("Planets with mass lower than Earth: ",below_earthmass)
print("Planets with mass higher than Earth: ",over_earthmass)

print("\n\nLowest Orb. Period [days]: ","\n")
print(df.loc[df["Orb. Period [days]"] == df["Orb. Period [days]"].min()],"\n")
print("Highest Orb. Period [days]: ","\n")
print(df.loc[df["Orb. Period [days]"] == df["Orb. Period [days]"].max()],"\n")

print("\nLowest mass: ","\n")
print(df.loc[df["Mass [Earth]"] == df["Mass [Earth]"].min()],"\n")
print("Highest mass: ","\n")
print(df.loc[df["Mass [Earth]"] == df["Mass [Earth]"].max()],"\n")


#print(df.corr())

##
## PLOTTING
##

def plotter(log_or_lin,var1,var2,hue):
    
    f, ax = plt.subplots(figsize=(8,8))
    ax.set_xscale(log_or_lin)
    ax.set_yscale(log_or_lin)

    sns.scatterplot(data=df, x=var1, y=var2,hue=hue, size=var2,)
    
    #sns.scatterplot(data=df_average, x="Orb. Period [days]", y="Mass [Earth]")

    #plt.text(av_orbper+10000, av_plmass+150, "Mean value", horizontalalignment='left', size='medium', color='black', weight='semibold', backgroundcolor='lightgray',alpha=.9)


    plt.show()

    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    #ax.set(ylabel="")
    sns.despine(trim=True, left=True)



for i in range(0, 6):
    print("\n\n Correlation between:", corrlist_indexes[i][0],"and",corrlist_indexes[i][1],"is", "{:.2f}".format(corrlist[i+1]*100),"%\n")
    plotter("log",corrlist_indexes[i][0],corrlist_indexes[i][1],"Star #")
