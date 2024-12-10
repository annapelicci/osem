import pyam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

### READ AND PROCESS COUNTRY-LEVEL POPULATION
### AND GDP FROM BASELINE SSP 

## read JSON iso2 to iso3 mapping
f = open("../SSP/iso3TEMBA.json")
iso2to3 = json.load(f)
f.close()

## read population per country file
pop = pd.read_excel('../SSP/population_country.xlsx', engine='openpyxl')
## read GDP per country file
gdp = pd.read_excel('../SSP/GDP_country.xlsx', engine='openpyxl')
## countries in MAF from SSP database website
maf = pd.read_csv('../SSP/MAF.csv', sep=', ', header=None, engine='python')
## read ISO mapping for all countries
countries = pd.read_excel('../SSP/ISO3166-1_codes_and_country_names.xlsx')

## POPULATION
#select countries in MAF
countries['MAF'] = countries['Countryname'].isin(maf.iloc[0].tolist())
countries = countries.loc[countries['MAF']==True]
## select pop data only for countries in MAF
pop = pop.loc[pop['Region'].isin(countries['ISO'])]
pop = pop.reset_index(drop=True)
## compute sum of all MAF countries pop
newrow = pop.iloc[0].copy()
newrow['Region'] = "R5.2MAF"
for ssp in pop['Scenario'].unique().tolist():
	newrow['Scenario'] = ssp
	sspselect = pop.loc[pop['Scenario']==ssp]
	for idx in range(5, len(pop.columns) - 1):
		newrow[idx] = sspselect[sspselect.columns[idx]].sum()
	# pop = pop.append(newrow)
	pop = pd.concat([pop,pd.DataFrame(newrow).T])
pop.to_excel('../SSP/population_country_MAF.xlsx', index=None)

## GDP
## select gdp data only for countries in MAF
gdp = gdp.loc[gdp['Region'].isin(countries['ISO'])]
gdp = gdp.reset_index(drop=True)
## compute sum of all MAF countries pop
newrow = gdp.iloc[0].copy()
newrow['Region'] = "R5.2MAF"
for ssp in gdp['Scenario'].unique().tolist():
	newrow['Scenario'] = ssp
	sspselect = gdp.loc[gdp['Scenario']==ssp]
	for idx in range(5, len(gdp.columns) - 1):
		newrow[idx] = sspselect[sspselect.columns[idx]].sum()
	# gdp = gdp.append(newrow)
	gdp = pd.concat([gdp,pd.DataFrame(newrow).T])

gdp.to_excel('../SSP/GDP_country_MAF.xlsx', index=None)

### READ ELECTRICITY DEMAND PROJECTIONS
dataset = 'ISIMIP2b' # baseline, SSP19, CMIP6, ISIMIP2b

## read data 
if dataset == 'ISIMIP2b':
	eleMAF = pd.read_excel('../SSP/ISIMIP2b/elec.xlsx', na_filter=False)		
eleMAF = eleMAF.iloc[:-2]
popMAF = pd.read_excel('../SSP/population_country_MAF.xlsx')
gdpMAF = pd.read_excel('../SSP/GDP_country_MAF.xlsx')

## read osemosys energy demand
if dataset == 'ISIMIP2b':
	TEMBARef = pd.read_excel('../jrc_temba-master/input_data/TEMBA_Refer.xlsx', 
		sheet_name='SpecifiedAnnualDemand', index_col='FUEL')
	TEMBA15 = pd.read_excel('../jrc_temba-master/input_data/TEMBA_1.5.xlsx', 
		sheet_name='SpecifiedAnnualDemand', index_col='FUEL')
	TEMBA20 = pd.read_excel('../jrc_temba-master/input_data/TEMBA_2.0.xlsx', 
		sheet_name='SpecifiedAnnualDemand', index_col='FUEL')
	scenarios = eleMAF['Scenario'].values


for ssp in scenarios:
	print(ssp)
	if dataset == 'ISIMIP2b':
		if ssp[-2:] == '26':
			TEMBA = TEMBA20
		else:
			TEMBA = TEMBARef
	scenario = pd.DataFrame(TEMBA[[x for x in range(2015,2021)]])
	subeleMAF = eleMAF.loc[eleMAF['Scenario']==ssp]

	for y in range(2021,2051):
		if y <= 2030:
			s_subeleMAF = subeleMAF[2020] * (2030-y)/(2030-2020) + \
				subeleMAF[2030] * (1 - (2030-y)/(2030-2020))
			gdpyear = gdpMAF.loc[gdpMAF['Scenario'].str.contains(ssp[:4])][['Region',2020,2030]]
			gdpyear[y] = gdpMAF.loc[gdpMAF['Scenario'].str.contains(ssp[:4])][2020] * (2030-y)/(2030-2020) + \
				gdpMAF.loc[gdpMAF['Scenario'].str.contains(ssp[:4])][2030] * (1 - (2030-y)/(2030-2020))
		elif y <= 2040:
			s_subeleMAF = subeleMAF[2030] * (2040-y)/(2040-2030) + \
				subeleMAF[2040] * (1 - (2040-y)/(2040-2030))
			gdpyear = gdpMAF.loc[gdpMAF['Scenario'].str.contains(ssp[:4])][['Region',2030,2040]]			
			gdpyear[y] = gdpMAF.loc[gdpMAF['Scenario'].str.contains(ssp[:4])][2030] * (2040-y)/(2040-2030) + \
				gdpMAF.loc[gdpMAF['Scenario'].str.contains(ssp[:4])][2040] * (1 - (2040-y)/(2040-2030))
		elif y <= 2050:
			s_subeleMAF = subeleMAF[2040] * (2050-y)/(2050-2040) + \
				subeleMAF[2050] * (1 - (2050-y)/(2050-2040))
			gdpyear = gdpMAF.loc[gdpMAF['Scenario'].str.contains(ssp[:4])][['Region',2040,2050]]
			gdpyear[y] = gdpMAF.loc[gdpMAF['Scenario'].str.contains(ssp[:4])][2040] * (2050-y)/(2050-2040) + \
				gdpMAF.loc[gdpMAF['Scenario'].str.contains(ssp[:4])][2050] * (1 - (2050-y)/(2050-2040))
		# gdpyear = gdpMAF.loc[gdpMAF['Scenario'].str.contains(ssp[:4])][['Region',y]]
		totgdpMAF = gdpyear.loc[gdpyear['Region']=='R5.2MAF'][y].values[0]
		weights = TEMBA.copy()
		vec = [gdpyear.loc[gdpyear['Region'].str.contains(iso2to3[el])][y].values/totgdpMAF if el!= 'SS' else np.array([0.0]) for el in [x[:2] for x in TEMBA.index]]
		weights['SSP'] = [x[0] for x in vec]
		# scenario[y] = TEMBA[y] * (2050-y)/(2050-2021) + \
		# 	s_subeleMAF.iloc[0]*1000 * weights['SSP'] * (1 - (2050-y)/(2050-2021))
		scenario[y] = TEMBA[y] * (2100-y)/(2100-2021) + \
			s_subeleMAF.iloc[0]*1000 * weights['SSP'] * (1 - (2100-y)/(2100-2021))
	scenario.iloc[scenario.index=='SSEL03'] = scenario.iloc[scenario.index=='SDEL03'] * \
		TEMBA.iloc[TEMBA.index=='SSEL03'].values[0][:2051-2015] / (TEMBA.iloc[TEMBA.index=='SDEL03'].values[0][:2051-2015] + TEMBA.iloc[TEMBA.index=='SSEL03'].values[0][:2051-2015])
	scenario.iloc[scenario.index=='SDEL03'] = scenario.iloc[scenario.index=='SDEL03'] * \
		TEMBA.iloc[TEMBA.index=='SDEL03'].values[0][:2051-2015] / (TEMBA.iloc[TEMBA.index=='SDEL03'].values[0][:2051-2015] + TEMBA.iloc[TEMBA.index=='SSEL03'].values[0][:2051-2015])
	scenario['SSP'] = ssp
	try:
		sspdwnscl = pd.concat([sspdwnscl, scenario])
		# sspdwnscl = sspdwnscl.append(scenario)
	except NameError:
		sspdwnscl = pd.DataFrame(scenario)

print(sspdwnscl)	
sortedcol = ['SSP']
[sortedcol.append(x) for x in range(2015,2051)]
sspdwnscl = sspdwnscl[sortedcol]
sspdwnscl.to_excel('../SSP/'+dataset+'/SpecifiedAnnualDemandSSP.xlsx')

country = 'LY'

fig, ax = plt.subplots()
for ssp in scenarios:
	select = sspdwnscl[sspdwnscl.index.str.contains(country)]
	select = select.loc[select['SSP'].str.contains(ssp)]
	select = select.T.drop('SSP')
	select[country+'EL03'].plot(ax = ax, label=ssp)

ax.plot([int(float(x)) for x in TEMBA20.columns], TEMBA20[TEMBA20.index==country+'EL03'].T.values, 'k--')
ax.plot([int(float(x)) for x in TEMBARef.columns], TEMBARef[TEMBARef.index==country+'EL03'].T.values, 'k--')
plt.legend()
plt.show()
