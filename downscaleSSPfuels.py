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

## read data 
# eleMAF = pd.read_excel('../SSP/electricity_MAF_MarkerSSP.xlsx', na_filter=False)
popMAF = pd.read_excel('../SSP/population_country_MAF.xlsx')
gdpMAF = pd.read_excel('../SSP/GDP_country_MAF.xlsx')

# first downscaling type: use temba to find average weight
# and then apply ssp gdp downscaling weight
### READ FUEL DEMAND PROJECTIONS
dataset = 'ISIMIP2b' # baseline, SSP19, CMIP6, ISIMIP2b

## read data 
if dataset == 'ISIMIP2b':
	TEMBARef = pd.read_excel('../jrc_temba-master/input_data/TEMBA_Refer.xlsx', 
		sheet_name='AccumulatedAnnualDemand', index_col='FUEL', usecols=[x for x in range(0,80)])
	TEMBA15 = pd.read_excel('../jrc_temba-master/input_data/TEMBA_1.5.xlsx', 
		sheet_name='AccumulatedAnnualDemand', index_col='FUEL', usecols=[x for x in range(0,80)])
	TEMBA20 = pd.read_excel('../jrc_temba-master/input_data/TEMBA_2.0.xlsx', 
		sheet_name='AccumulatedAnnualDemand', index_col='FUEL', usecols=[x for x in range(0,80)])
	eleMAF = pd.read_excel('../SSP/ISIMIP2b/elec.xlsx', na_filter=False)	
	eleMAF = eleMAF.iloc[:-2]
	scenarios = eleMAF['Scenario'].values

fueldict = {'COAL':['CO','CH'],'BIOMASS':['BO','FW'],'HEAT':['HE'],'GASES':['GA'],'LIQUIDS':['CR','HF','LF']}

# TEMBA['Fuel'] = [x[2:4] for x in TEMBA.index]
# TEMBA['Country'] = [x[:2] for x in TEMBA.index]

for ssp in scenarios:
	print(ssp)
	for key in fueldict:
		print(key)
		if dataset == 'ISIMIP2b':
			if ssp[-2:] == '26':
				TEMBA = TEMBA20
			else:
				TEMBA = TEMBARef

		TEMBA['Fuel'] = [x[2:4] for x in TEMBA.index]
		TEMBA['Country'] = [x[:2] for x in TEMBA.index]
		TEMBAf = TEMBA.loc[TEMBA['Fuel'].str.contains('|'.join(fueldict[key]))]
		# TEMBAf = TEMBAf.drop('fuel', axis=1)
		# TEMBAf = TEMBAf.loc[~TEMBAf.index.str.contains('EX')]
		if '5' not in ssp  and key=='HEAT':
			scenario = pd.DataFrame(TEMBAf[[x for x in range(2015,2051)]])
			scenario['SSP'] = ssp
			sortedcols = ['SSP']
			[sortedcols.append(x) for x in range(2015,2051)]
			scenario = scenario[sortedcols]
			sspdwnscl = pd.concat([sspdwnscl,scenario])
			# sspdwnscl = sspdwnscl.append(scenario)
			continue
		scenario = pd.DataFrame(TEMBAf[[x for x in range(2015,2021)]])
		SSPf = pd.read_excel('../SSP/'+dataset+'/'+key.lower()+'.xlsx', na_filter=False)
		SSPf = SSPf.iloc[:-2]
		subfMAF = SSPf.loc[SSPf['Scenario']==ssp]
		count = 1
		for y in range(2021,2051):
			if y <= 2030:
				s_subfMAF = subfMAF[2020] * (2030-y)/(2030-2020) + \
					subfMAF[2030] * (1 - (2030-y)/(2030-2020))
				gdpyear = gdpMAF.loc[gdpMAF['Scenario'].str.contains(ssp[:4])][['Region',2020,2030]]
				gdpyear[y] = gdpMAF.loc[gdpMAF['Scenario'].str.contains(ssp[:4])][2020] * (2030-y)/(2030-2020) + \
					gdpMAF.loc[gdpMAF['Scenario'].str.contains(ssp[:4])][2030] * (1 - (2030-y)/(2030-2020))
			elif y <= 2040:
				s_subfMAF = subfMAF[2030] * (2040-y)/(2040-2030) + \
					subfMAF[2040] * (1 - (2040-y)/(2040-2030))
				gdpyear = gdpMAF.loc[gdpMAF['Scenario'].str.contains(ssp[:4])][['Region',2030,2040]]			
				gdpyear[y] = gdpMAF.loc[gdpMAF['Scenario'].str.contains(ssp[:4])][2030] * (2040-y)/(2040-2030) + \
					gdpMAF.loc[gdpMAF['Scenario'].str.contains(ssp[:4])][2040] * (1 - (2040-y)/(2040-2030))
			elif y <= 2050:
				s_subfMAF = subfMAF[2040] * (2050-y)/(2050-2040) + \
					subfMAF[2050] * (1 - (2050-y)/(2050-2040))
				gdpyear = gdpMAF.loc[gdpMAF['Scenario'].str.contains(ssp[:4])][['Region',2040,2050]]
				gdpyear[y] = gdpMAF.loc[gdpMAF['Scenario'].str.contains(ssp[:4])][2040] * (2050-y)/(2050-2040) + \
					gdpMAF.loc[gdpMAF['Scenario'].str.contains(ssp[:4])][2050] * (1 - (2050-y)/(2050-2040))
			totgdpMAF = gdpyear.loc[gdpyear['Region']=='R5.2MAF'][y].values[0]
			weights = TEMBAf.copy()
			vec = [gdpyear.loc[gdpyear['Region'].str.contains(iso2to3[el])][y].values/totgdpMAF if el!= 'SS' else np.array([0.0]) for el in [x[:2] for x in TEMBAf.index]]
			weights['SSP'] = [x[0] for x in vec]
			scenario[y] = (TEMBAf[y] * (2100-y)/(2100-2021) + \
				s_subfMAF.iloc[0]*1000 * weights['SSP'] * (1 - (2100-y)/(2100-2021)) )*\
				(TEMBAf[y]>0).astype(int)

			if key == 'LIQUIDS':
				for cntr in TEMBAf['Country'].unique():
					weight_fix = TEMBAf.loc[TEMBAf.index.str.contains(cntr+'HF')][y].values / TEMBAf.loc[TEMBAf.index.str.contains(cntr+'LF')][y].values
					scenario.loc[scenario.index.str.contains(cntr+'HF'), y] = scenario.loc[scenario.index.str.contains(cntr+'LF')][y].values * weight_fix[0]
				if ssp[-2:] in ['19','26']:
					scenario.loc[scenario.index.str.contains('EGHF'), y] = TEMBAf.loc[TEMBAf.index.str.contains('EGHF')][y].values
					scenario.loc[scenario.index.str.contains('EGLF'), y] = TEMBAf.loc[TEMBAf.index.str.contains('EGLF')][y].values
			if key=='GASES':
				scenario.loc[scenario.index.str.contains('MAGA'), y] = TEMBAf.loc[TEMBAf.index.str.contains('MAGA')][y].values
				scenario.loc[scenario.index.str.contains('TZGA'), y] = TEMBAf.loc[TEMBAf.index.str.contains('TZGA')][y].values
				if ssp[-2:] in ['19','26']:
					scenario.loc[scenario.index.str.contains('EGGA'), y] = TEMBAf.loc[TEMBAf.index.str.contains('EGGA')][y].values
			if key=='BIOMASS':
				scenario.loc[scenario.index.str.contains('MAFW'), y] = TEMBAf.loc[TEMBAf.index.str.contains('MAFW')][y].values
				scenario.loc[scenario.index.str.contains('EGFW'), y] = TEMBAf.loc[TEMBAf.index.str.contains('EGFW')][y].values
				scenario.loc[scenario.index.str.contains('DZFW'), y] = TEMBAf.loc[TEMBAf.index.str.contains('DZFW')][y].values
			if key=='COAL':
				scenario.loc[scenario.index.str.contains('EGCO'), y] = TEMBAf.loc[TEMBAf.index.str.contains('EGCO')][y].values
			scenario.loc[scenario.index.str.contains('EX'), y] = TEMBAf.loc[TEMBAf.index.str.contains('EX'), y].values
		sudan_corr = TEMBAf.iloc[TEMBAf.index.str.contains('SD')]
		indexes = []
		for el in sudan_corr.index:
			if '3X' in el:
				indexes.append(el[2:])
		for el in indexes:
			if all(x > 0 for x in TEMBAf.iloc[TEMBAf.index.str.contains('SD'+el)].values[0][:2051-2015]):
				scenario.iloc[scenario.index.str.contains('SS'+el)] = scenario.iloc[scenario.index.str.contains('SD'+el)] * \
					TEMBAf.iloc[TEMBAf.index.str.contains('SS'+el)].values[0][:2051-2015] / (TEMBAf.iloc[TEMBAf.index.str.contains('SD'+el)].values[0][:2051-2015] + TEMBAf.iloc[TEMBAf.index.str.contains('SD'+el)].values[0][:2051-2015])
		scenario['SSP'] = ssp

		try:
			sspdwnscl = pd.concat([sspdwnscl, scenario])
		except NameError:
			sspdwnscl = pd.DataFrame(scenario)

sortedcol = ['SSP']
[sortedcol.append(x) for x in range(2015,2051)]
sspdwnscl = sspdwnscl[sortedcol]
sspdwnscl.to_excel('../SSP/'+dataset+'/AccumulatedAnnualDemandSSP.xlsx')
TEMBA = TEMBA.drop(['Fuel','Country'], axis=1)
TEMBARef = TEMBARef.drop(['Fuel','Country'], axis=1)
TEMBA20 = TEMBA20.drop(['Fuel','Country'], axis=1)
for el in TEMBA.index:
	if el[:2] in ['TZ']:
		fig, ax = plt.subplots()
		country = el
		for ssp in scenarios:
			select = sspdwnscl[sspdwnscl.index.str.contains(country)]
			select = select.loc[select['SSP'].str.contains(ssp)]
			select = select.T.drop('SSP')
			select[country].plot(ax = ax, label=ssp)
		ax.plot([int(float(x)) for x in TEMBARef.columns], TEMBARef[TEMBARef.index==country].T.values, 'k--', label='TEMBARef')
		ax.plot([int(float(x)) for x in TEMBA20.columns], TEMBA20[TEMBA20.index==country].T.values, 'k--', label='TEMBA20')
		ax.plot([int(float(x)) for x in TEMBA15.columns], TEMBA15[TEMBA15.index==country].T.values, 'k--', label='TEMBA15')
		ax.set_title(country)
		ax.legend()
plt.show()