import pandas as pd
import numpy as np
import sys

PP_lines_all = pd.read_csv('../jrc_temba-master/input_data/techcodes.csv')
PP_lines_all = PP_lines_all.loc[PP_lines_all['tech_name'].str.contains('EL existing trade link|EL planned trade link'),'tech_code'].values


def prepare_scenarios(year_start, year_end, dataset, set_capacity, 
	capacity_scenario, set_hydro_capacity_only, hydro_scenario,
	first_hydro_year=2022, TLines=True, hydrology=None):

	PP_lines = ['LYELEGBP00','CDELBIBP00','CDELRWBP00',
		'CDELZABP00','CDELZMBP00','AOELCDBP00','MWELTZBP00',
		'TZELZMBP00','ZMELTZBP00','SNELMRBP00']


	if set_hydro_capacity_only is True and set_capacity is True:
		print('Error: please set True either set_hydro_capacity_only or set_capacity') 
		sys.exit(1)

	## read AHA
	AHA = pd.read_excel('../AHA/African_Hydropower_Atlas_v2-0_PoliTechM.xlsx', 
		sheet_name=1, header=2, usecols=[x for x in range(1,50)])
	AHAcapF = pd.read_excel('../AHA/African_Hydropower_Atlas_v2-0_PoliTechM.xlsx', 
		sheet_name=4, header=0, usecols=[x for x in range(0,50)])
	AHAcapF = AHAcapF.dropna(axis=0, how='any')
	AHAcapF_SSP126 = pd.read_excel('../AHA/African_Hydropower_Atlas_v2-0_PoliTechM.xlsx', 
		sheet_name=5, header=0, usecols=[x for x in range(0,50)])
	AHAcapF_SSP126 = AHAcapF_SSP126.dropna(axis=0, how='any')
	AHAcapF_SSP460 = pd.read_excel('../AHA/African_Hydropower_Atlas_v2-0_PoliTechM.xlsx', 
		sheet_name=6, header=0, usecols=[x for x in range(0,50)])
	AHAcapF_SSP460 = AHAcapF_SSP460.dropna(axis=0, how='any')
	AHAcapF_SSP585 = pd.read_excel('../AHA/African_Hydropower_Atlas_v2-0_PoliTechM.xlsx', 
		sheet_name=7, header=0, usecols=[x for x in range(0,50)])
	AHAcapF_SSP585 = AHAcapF_SSP585.dropna(axis=0, how='any')

	## associate OSeMOSYS-TEMBA coherent country code to each hydropower unit in AHA
	countries = pd.read_csv("../jrc_temba-master/input_data/countrycode.csv")
	countries['Country_comp'] = countries['Country Name'].str.upper()

	## some countries have a slighlty different name
	## in AHA and OSeMOSYS-TEMBA, they are corrected
	## Madagascar, not considered in OSeMOSYS-TEMBA is removed
	countries_correct = {"CONGO DEMOCRATIC REPUBLIC": "DRC", 
		"CONGO PEOPLE REPUBLIC": "CONGO", "SWAZILAND": "ESWATINI",
		"COTE D'IVOIRE": "CÔTE D'IVOIRE"}
	for el in countries_correct.keys():
		countries.loc[countries['Country_comp'] == el, 
		'Country_comp'] = countries_correct[el]

	## add country code form OSeMOSYS-TEMBA in AHA
	result = pd.merge(AHA, countries, 
		left_on='Country', right_on='Country_comp')
	AHA = result.drop(columns=['Country Name',
		'Country_comp']).copy()
	result = pd.merge(AHAcapF, countries, 
		left_on='Country', right_on='Country_comp')
	AHAcapF = result.drop(columns=['Country Name',
		'Country_comp']).copy()
	result = pd.merge(AHAcapF_SSP126, countries, 
		left_on='Country', right_on='Country_comp')
	AHAcapF_SSP126 = result.drop(columns=['Country Name',
		'Country_comp']).copy()
	result = pd.merge(AHAcapF_SSP460, countries, 
		left_on='Country', right_on='Country_comp')
	AHAcapF_SSP460 = result.drop(columns=['Country Name',
		'Country_comp']).copy()
	result = pd.merge(AHAcapF_SSP585, countries, 
		left_on='Country', right_on='Country_comp')
	AHAcapF_SSP585 = result.drop(columns=['Country Name',
		'Country_comp']).copy()

	## clean up unit names
	AHA['TEMBANAME'] = AHA['Country code'] + AHA['Unit Name'].str.replace(' |\(|\)|\'|\/|\.|\-|\,','', regex=True).str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
	AHAcapF['TEMBANAME'] = AHAcapF['Country code'] + AHAcapF['Name'].str.replace(' |\(|\)|\'|\/|\.|\-|\,','', regex=True).str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
	AHAcapF_SSP126['TEMBANAME'] = AHAcapF_SSP126['Country code'] + AHAcapF_SSP126['Name'].str.replace(' |\(|\)|\'|\/|\.|\-|\,','', regex=True).str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
	AHAcapF_SSP460['TEMBANAME'] = AHAcapF_SSP460['Country code'] + AHAcapF_SSP460['Name'].str.replace(' |\(|\)|\'|\/|\.|\-|\,','', regex=True).str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
	AHAcapF_SSP585['TEMBANAME'] = AHAcapF_SSP585['Country code'] + AHAcapF_SSP585['Name'].str.replace(' |\(|\)|\'|\/|\.|\-|\,','', regex=True).str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
	## the following unit appears two times
	AHA.loc[(AHA['TEMBANAME']=='GQLofa') & (AHA['Status']=='Candidate'),'TEMBANAME'] = "GQLofa2"
	## set NAN to zero reservoir volume: run-of-river hydro
	AHA.loc[AHA['Reservoir Size'].isna(),'Reservoir Size'] = 0.0

	## create three dataframes: ex_res, ex_ror, planned
	ex_res = AHA.loc[(AHA['Reservoir Size']>0.0) & (AHA['Status']=='Existing')]
	ex_ror = AHA.loc[(AHA['Reservoir Size']==0.0) & (AHA['Status']=='Existing')]
	fut = AHA.loc[~(AHA['Status']=='Existing')]

	if dataset=='CMIP6':
		eleMAF = pd.read_excel('../SSP/CMIP6/elec.xlsx', na_filter=False)
		eleMAF = eleMAF.iloc[:-2]
		scenarios = eleMAF['Scenario'].unique()
		print(scenarios)
	if dataset=='ISIMIP2b':
		eleMAF = pd.read_excel('../SSP/ISIMIP2b/elec.xlsx', na_filter=False)
		eleMAF = eleMAF.iloc[:-2]
		scenarios = eleMAF['Scenario'].sort_values().unique()
		print(scenarios)

	## reading TEMBA FILES
	TEMBARef = pd.ExcelFile('../jrc_temba-master/input_data/TEMBA_Refer.xlsx')
	TEMBA15 = pd.ExcelFile('../jrc_temba-master/input_data/TEMBA_1.5.xlsx')
	TEMBA20 = pd.ExcelFile('../jrc_temba-master/input_data/TEMBA_2.0.xlsx')

	for scenario in scenarios:
		## read TEMBA original input data
		if '19' in scenario:
			TEMBAExcel = TEMBA15
		elif '26' in scenario:
			TEMBAExcel = TEMBA20
		else:
			TEMBAExcel = TEMBARef
		if 'robust' in scenario:
			TEMBAExcel = pd.ExcelFile("../jrc_temba-master/input_data/TEMBA_Refer_scenario.xlsx")
		## create writer to write modified TEMBA input data
		name_scenario = scenario
		if set_capacity is True:
			name_scenario += capacity_scenario+'_fixed_cap'
		elif set_hydro_capacity_only is True:
			name_scenario += '_AHAplan'
		if hydro_scenario=='dry':
			name_scenario += '_'+hydro_scenario
		if hydrology is not None:
			name_scenario += '_' + hydrology
		if TLines==False:
			name_scenario += '_noTL'
		writer = pd.ExcelWriter("../jrc_temba-master/input_data/TEMBA_"+name_scenario+".xlsx")

		## list of countries for which 
		## hydropower technology in TEMBA has to be updated
		## using data from AHA
		hydms03 = [x+"HYDMS03X" for x in ex_res['Country code'].unique().tolist()]
		hydms02 = [x+"HYDMS02X" for x in ex_ror['Country code'].unique().tolist()]
		allcountries = [x+"HYDMS03X" for x in AHA['Country code'].unique().tolist()]

		## save here new capacity to be assigned to hydms0X techs
		newcap = {}

		for sheet in TEMBAExcel.sheet_names:
			print(sheet)
			df = pd.read_excel(TEMBAExcel, sheet, index_col=None, header=None)
			## modify technology sheet
			if sheet=='TECHNOLOGY':
				df = pd.concat([df, AHA['TEMBANAME']])
				techslist = df[0].values
				TLinesList = [x for x in df[0].values if x[2:4]=='EL' and x[6:]=='BP00']
			## modify technology sheet
			if sheet=='YEAR' or sheet=='COMMONCAPYEARS':
				df = df.drop(df.index[df[0]>year_end])
				df = df.drop(df.index[df[0]<year_start])
			sheets = ['AvailabilityFactor', 
				'CapacityToActivityUnit','EmissionActivityRatio',
				'CapitalCost','FixedCost','InputActivityRatio',
				'OutputActivityRatio','OperationalLife',
				'VariableCost']
			if sheet in sheets:
				for x in allcountries:
					idx = df.index[df[0]==x]
					select = df.iloc[idx].copy()
					dams = AHA.loc[AHA['Country code']==x[:2]]['TEMBANAME'].tolist()
					for dam in dams:
						newrow = select.copy()
						newrow[0] = dam
						## the following are to avoid degeneracy using data from IRENA renewable cost database 2020
						if sheet in ['CapitalCost']:
							if AHA.loc[AHA['TEMBANAME']==dam]['Capacity'].values[0] > 10 and AHA.loc[AHA['TEMBANAME']==dam]['Capacity'].values[0]<500:
								newrow[[x for x in range(1,57)]] = 2836.5 - (2836.5 - 2445.5)*(AHA.loc[AHA['TEMBANAME']==dam]['Capacity'].values[0] - 10) / (500 - 10)
							elif AHA.loc[AHA['TEMBANAME']==dam]['Capacity'].values[0]>=500:
								newrow[[x for x in range(1,57)]] = 2445.5 - (2445.5 - 2054.5)*(AHA.loc[AHA['TEMBANAME']==dam]['Capacity'].values[0] - 500) / (11000 - 500)						
							else:
								newrow[[x for x in range(1,57)]] = 3744.4 - (3744.4 - 2836.5)*(AHA.loc[AHA['TEMBANAME']==dam]['Capacity'].values[0] - 0.1) / 10
						if isinstance(newrow, pd.core.series.Series):
							df = pd.concat([df, pd.DataFrame(newrow).T])
						else:
							df = pd.concat([df, newrow])
					df.reset_index(drop=True)

			if sheet in ['CapitalCost']:
				df.loc[df[0].str.contains('BACKSTOP'), [x for x in range(1,57)]] = 0.0

			## update capacity factors using AHA
			if sheet in ['CapacityFactor']:
				for x in allcountries:
					idx = df.index[df[0]==x]
					select = df.iloc[idx].copy()
					dams = AHA.loc[AHA['Country code']==x[:2]]['TEMBANAME'].tolist()
					for dam in dams:
						newrow = select.copy()
						newrow[0] = dam
						if ((hydrology=='control') or (dataset=='CMIP6')) and dam in AHAcapF['TEMBANAME'].values:
							S1m = ['March_capacity_factor_'+hydro_scenario,'April_capacity_factor_'+hydro_scenario,'May_capacity_factor_'+hydro_scenario]
							newrow.loc[newrow[1].str.contains('S1'), [idx for idx in range(2,58)]] = AHAcapF.loc[AHAcapF['TEMBANAME']==dam, S1m].mean(axis=1).values[0]
							S2m = ['June_capacity_factor_'+hydro_scenario,'July_capacity_factor_'+hydro_scenario,'August_capacity_factor_'+hydro_scenario]
							newrow.loc[newrow[1].str.contains('S2'), [idx for idx in range(2,58)]] = AHAcapF.loc[AHAcapF['TEMBANAME']==dam, S2m].mean(axis=1).values[0]
							S3m = ['September_capacity_factor_'+hydro_scenario,'October_capacity_factor_'+hydro_scenario,'November_capacity_factor_'+hydro_scenario]
							newrow.loc[newrow[1].str.contains('S3'), [idx for idx in range(2,58)]] = AHAcapF.loc[AHAcapF['TEMBANAME']==dam, S3m].mean(axis=1).values[0]
							S4m = ['December_capacity_factor_'+hydro_scenario,'January_capacity_factor_'+hydro_scenario,'February_capacity_factor_'+hydro_scenario]
							newrow.loc[newrow[1].str.contains('S4'), [idx for idx in range(2,58)]] = AHAcapF.loc[AHAcapF['TEMBANAME']==dam, S4m].mean(axis=1).values[0]
						elif dataset=='ISIMIP2b' and scenario=='SSP1-26' and dam in AHAcapF['TEMBANAME'].values:
							S1m = ['March_capacity_factor_'+hydro_scenario,'April_capacity_factor_'+hydro_scenario,'May_capacity_factor_'+hydro_scenario]
							newrow.loc[newrow[1].str.contains('S1'), [idx for idx in range(2,58)]] = AHAcapF_SSP126.loc[AHAcapF_SSP126['TEMBANAME']==dam, S1m].mean(axis=1).values[0]
							S2m = ['June_capacity_factor_'+hydro_scenario,'July_capacity_factor_'+hydro_scenario,'August_capacity_factor_'+hydro_scenario]
							newrow.loc[newrow[1].str.contains('S2'), [idx for idx in range(2,58)]] = AHAcapF_SSP126.loc[AHAcapF_SSP126['TEMBANAME']==dam, S2m].mean(axis=1).values[0]
							S3m = ['September_capacity_factor_'+hydro_scenario,'October_capacity_factor_'+hydro_scenario,'November_capacity_factor_'+hydro_scenario]
							newrow.loc[newrow[1].str.contains('S3'), [idx for idx in range(2,58)]] = AHAcapF_SSP126.loc[AHAcapF_SSP126['TEMBANAME']==dam, S3m].mean(axis=1).values[0]
							S4m = ['December_capacity_factor_'+hydro_scenario,'January_capacity_factor_'+hydro_scenario,'February_capacity_factor_'+hydro_scenario]
							newrow.loc[newrow[1].str.contains('S4'), [idx for idx in range(2,58)]] = AHAcapF_SSP126.loc[AHAcapF_SSP126['TEMBANAME']==dam, S4m].mean(axis=1).values[0]
						elif dataset=='ISIMIP2b' and scenario=='SSP4-60' and dam in AHAcapF['TEMBANAME'].values:
							S1m = ['March_capacity_factor_'+hydro_scenario,'April_capacity_factor_'+hydro_scenario,'May_capacity_factor_'+hydro_scenario]
							newrow.loc[newrow[1].str.contains('S1'), [idx for idx in range(2,58)]] = AHAcapF_SSP460.loc[AHAcapF_SSP460['TEMBANAME']==dam, S1m].mean(axis=1).values[0]
							S2m = ['June_capacity_factor_'+hydro_scenario,'July_capacity_factor_'+hydro_scenario,'August_capacity_factor_'+hydro_scenario]
							newrow.loc[newrow[1].str.contains('S2'), [idx for idx in range(2,58)]] = AHAcapF_SSP460.loc[AHAcapF_SSP460['TEMBANAME']==dam, S2m].mean(axis=1).values[0]
							S3m = ['September_capacity_factor_'+hydro_scenario,'October_capacity_factor_'+hydro_scenario,'November_capacity_factor_'+hydro_scenario]
							newrow.loc[newrow[1].str.contains('S3'), [idx for idx in range(2,58)]] = AHAcapF_SSP460.loc[AHAcapF_SSP460['TEMBANAME']==dam, S3m].mean(axis=1).values[0]
							S4m = ['December_capacity_factor_'+hydro_scenario,'January_capacity_factor_'+hydro_scenario,'February_capacity_factor_'+hydro_scenario]
							newrow.loc[newrow[1].str.contains('S4'), [idx for idx in range(2,58)]] = AHAcapF_SSP460.loc[AHAcapF_SSP460['TEMBANAME']==dam, S4m].mean(axis=1).values[0]
						elif dataset=='ISIMIP2b' and scenario=='SSP5-Baseline' and dam in AHAcapF['TEMBANAME'].values:
							S1m = ['March_capacity_factor_'+hydro_scenario,'April_capacity_factor_'+hydro_scenario,'May_capacity_factor_'+hydro_scenario]
							newrow.loc[newrow[1].str.contains('S1'), [idx for idx in range(2,58)]] = AHAcapF_SSP585.loc[AHAcapF_SSP585['TEMBANAME']==dam, S1m].mean(axis=1).values[0]
							S2m = ['June_capacity_factor_'+hydro_scenario,'July_capacity_factor_'+hydro_scenario,'August_capacity_factor_'+hydro_scenario]
							newrow.loc[newrow[1].str.contains('S2'), [idx for idx in range(2,58)]] = AHAcapF_SSP585.loc[AHAcapF_SSP585['TEMBANAME']==dam, S2m].mean(axis=1).values[0]
							S3m = ['September_capacity_factor_'+hydro_scenario,'October_capacity_factor_'+hydro_scenario,'November_capacity_factor_'+hydro_scenario]
							newrow.loc[newrow[1].str.contains('S3'), [idx for idx in range(2,58)]] = AHAcapF_SSP585.loc[AHAcapF_SSP585['TEMBANAME']==dam, S3m].mean(axis=1).values[0]
							S4m = ['December_capacity_factor_'+hydro_scenario,'January_capacity_factor_'+hydro_scenario,'February_capacity_factor_'+hydro_scenario]
							newrow.loc[newrow[1].str.contains('S4'), [idx for idx in range(2,58)]] = AHAcapF_SSP585.loc[AHAcapF_SSP585['TEMBANAME']==dam, S4m].mean(axis=1).values[0]
						if isinstance(newrow, pd.core.series.Series):
							df = pd.concat([df, pd.DataFrame(newrow).T])
						else:
							df = pd.concat([df, newrow])
					df.reset_index(drop=True)		
			## update residual capacity based on AHA
			if sheet in ['ResidualCapacity']:
				for x in hydms03:
					idx = df.index[df[0]==x]
					select = df.iloc[idx].copy()
					select[[idx for idx in range(1, 57)]] = 0.0
					dams = ex_res.loc[ex_res['Country code']==x[:2]]['TEMBANAME'].tolist()
					count = 0.0
					## add hydropower units in the country
					for dam in dams:
						newrow = select.copy()
						newrow[0] = dam
						count += 0.001*ex_res.loc[ex_res['TEMBANAME']==dam]['Capacity'].values[0]
						installed_after_2015 = max(0, ex_res.loc[ex_res['TEMBANAME']==dam]['First Year'].values[0] - 2015)
						newrow[[idx for idx in range(1 + installed_after_2015, 57)]] = ex_res.loc[ex_res['TEMBANAME']==dam]['Capacity'].values[0]/1000.0
						if isinstance(newrow, pd.core.series.Series):
							df = pd.concat([df, pd.DataFrame(newrow).T])
						else:
							df = pd.concat([df, newrow])
					## update hydms03 tech capacity
					df.iloc[idx,[idx for idx in range(1,57)]] = np.max([0.0, df.iloc[idx][5].values[0] - count ])
					newcap[x] = df.iloc[idx][5].values[0]
					df.reset_index(drop=True)
				for x in hydms02:
					idx = df.index[df[0]==x]
					select = df.iloc[idx].copy()
					select[[idx for idx in range(1, 57)]] = 0.0
					dams = ex_ror.loc[ex_ror['Country code']==x[:2]]['TEMBANAME'].tolist()
					count = 0.0
					## add hydropower units in the country
					for dam in dams:
						newrow = select.copy()
						newrow[0] = dam
						count += 0.001*ex_ror.loc[ex_ror['TEMBANAME']==dam]['Capacity'].values[0]
						installed_after_2015 = max(0, ex_ror.loc[ex_ror['TEMBANAME']==dam]['First Year'].values[0] - 2015)
						newrow[[idx for idx in range(1 + installed_after_2015 ,57)]] = ex_ror.loc[ex_ror['TEMBANAME']==dam]['Capacity'].values[0]/1000.0 
						if isinstance(newrow, pd.core.series.Series):
							df = pd.concat([df, pd.DataFrame(newrow).T])
						else:
							df = pd.concat([df, newrow])
					## update hydms03 tech capacity
					mincap = 0.0
					df.iloc[idx,[idx for idx in range(1,57)]] = np.max([0.0, df.iloc[idx][5].values[0] - count + mincap])
					newcap[x] = df.iloc[idx][5].values[0]
					df.reset_index(drop=True)
				select = df.iloc[0].copy()
				## add future poewr plant residual capacity (equal to 0)
				for x in fut['TEMBANAME'].tolist():
					newrow = select.copy()
					newrow[0] = x
					newrow[[idx for idx in range(1,57)]] = 0.0
					if isinstance(newrow, pd.core.series.Series):
						df = pd.concat([df, pd.DataFrame(newrow).T])
					else:
						df = pd.concat([df, newrow])
			## update annual max capacity according to above
			if sheet in ['TotalAnnualMaxCapacity'] and sheet not in ['TotalAnnualMaxCapacityInvestmen']:
				for x in hydms03:
					df.loc[df[0]==x, [idx for idx in range(1,57)]] = newcap[x]
				for x in hydms02:
					df.loc[df[0]==x, [idx for idx in range(1,57)]] = newcap[x]
				select = df.iloc[0].copy()
				for x in ex_res['TEMBANAME'].tolist():
					select[0] = x
					select[[idx for idx in range(1,57)]] = AHA.loc[AHA['TEMBANAME']==x]['Capacity'].values[0]/1000.0
					if isinstance(select, pd.core.series.Series):
						df = pd.concat([df, pd.DataFrame(select).T])
					else:
						df = pd.concat([df, select])
				for x in ex_ror['TEMBANAME'].tolist():
					select[0] = x
					select[[idx for idx in range(1,57)]] = AHA.loc[AHA['TEMBANAME']==x]['Capacity'].values[0]/1000.0
					if isinstance(select, pd.core.series.Series):
						df = pd.concat([df, pd.DataFrame(select).T])
					else:
						df = pd.concat([df, select])
				for x in fut['TEMBANAME'].tolist():
					select[0] = x
					select[[idx for idx in range(1,57)]] = AHA.loc[AHA['TEMBANAME']==x]['Capacity'].values[0]/1000.0
					if isinstance(select, pd.core.series.Series):
						df = pd.concat([df, pd.DataFrame(select).T])
					else:
						df = pd.concat([df, select])
			if sheet in ['TotalAnnualMaxCapacityInvestmen']:
				if set_capacity is not True:
					for x in allcountries:
						if not(df.loc[df[0]==x].empty):
							df.loc[df[0]==x, [idx for idx in range(1,57)]] = 0.0
						else:
							newrow = df.iloc[1].copy()
							newrow[0] = x
							newrow[[x for x in range(1,57)]] = 0.0
							if isinstance(newrow, pd.core.series.Series):
								df = pd.concat([df, pd.DataFrame(newrow).T])
							else:
								df = pd.concat([df, newrow])
					for x in allcountries:
						x = x[:-2]+'2X'
						if not(df.loc[df[0]==x].empty):
							df.loc[df[0]==x, [idx for idx in range(1,57)]] = 0.0
						else:
							newrow = df.iloc[1].copy()
							newrow[0] = x
							newrow[[x for x in range(1,57)]] = 0.0
							if isinstance(newrow, pd.core.series.Series):
								df = pd.concat([df, pd.DataFrame(newrow).T])
							else:
								df = pd.concat([df, newrow])
					select = df.iloc[0].copy()
					for x in ex_res['TEMBANAME'].tolist():
						select[0] = x
						select[[idx for idx in range(1,57)]] = 0.0
						if isinstance(select, pd.core.series.Series):
							df = pd.concat([df, pd.DataFrame(select).T])
						else:
							df = pd.concat([df, select])
					for x in ex_ror['TEMBANAME'].tolist():
						select[0] = x
						select[[idx for idx in range(1,57)]] = 0.0
						if isinstance(select, pd.core.series.Series):
							df = pd.concat([df, pd.DataFrame(select).T])
						else:
							df = pd.concat([df, select])
					for x in fut['TEMBANAME'].tolist():
						select[0] = x
						select[[idx for idx in range(1,first_hydro_year-year_start+1)]] = 0.0
						select[[idx for idx in range(1+first_hydro_year-year_start,57)]] = AHA.loc[AHA['TEMBANAME']==x]['Capacity'].values[0]/1000.0
						if 'Renaissance' in x:
							select[[idx for idx in range(1,2023-year_start+1) ]] = 0.0
							select[2023-year_start+1] = AHA.loc[AHA['TEMBANAME']==x]['Capacity'].values[0]/1000.0
							select[[idx for idx in range(2023-year_start+2,57) ]] = 0.0
						if set_hydro_capacity_only is True:
							select[[idx for idx in range(1,57)]] = 0.0
							select[[AHA.loc[AHA['TEMBANAME']==x]['First Year'].values[0] - year_start + 1]] = AHA.loc[AHA['TEMBANAME']==x]['Capacity'].values[0]/1000.0

						if isinstance(select, pd.core.series.Series):
							df = pd.concat([df, pd.DataFrame(select).T])
						else:
							df = pd.concat([df, select])
				elif set_capacity is True:
					with open('../jrc_temba-master/output_data/sorted_TEMBA_'+scenario+capacity_scenario+'.txt') as f:
						file = f.read().split("\n")[:-1]
					data = []
					for el in file:
						data.append(el.split('\t'))
					select = []
					for row in data:
						if row[0]=='NewCapacity':
							select.append(row)
					select = pd.DataFrame(select)
					for tech in select[2]:
						if df.loc[df[0]==tech].empty and 'BACKSTOP' not in tech:
							newrow = df.iloc[1].copy()
							newrow[0] = tech
							newrow.loc[[x for x in range(year_start - 2015 + 1, year_end - 2015 + 1 + 1)]] = select.loc[select[2]==tech,[x for x in range(3,3+year_end - year_start + 1)]].values[0]
							if isinstance(newrow, pd.core.series.Series):
								df = pd.concat([df, pd.DataFrame(newrow).T])
							else:
								df = pd.concat([df, newrow])
						else:
							df.loc[df[0]==tech, [x for x in range(year_start - 2015 + 1, year_end - 2015 + 1 + 1)]] = select.loc[select[2]==tech,[x for x in range(3,3+year_end - year_start + 1)]].values[0]	
			if sheet in ['TotalAnnualMinCapacityInvestmen']:
				if set_capacity is not True:
					for x in allcountries:
						if not(df.loc[df[0]==x].empty):
							df.loc[df[0]==x, [idx for idx in range(1,57)]] = 0.0
						else:
							newrow = df.iloc[1].copy()
							newrow[0] = x
							newrow[[x for x in range(1,57)]] = 0.0
							# df = df.append(select)
							if isinstance(newrow, pd.core.series.Series):
								df = pd.concat([df, pd.DataFrame(newrow).T])
							else:
								df = pd.concat([df, newrow])
					for x in allcountries:
						x = x[:-2]+'2X'
						if not(df.loc[df[0]==x].empty):
							df.loc[df[0]==x, [idx for idx in range(1,57)]] = 0.0
						else:
							newrow = df.iloc[1].copy()
							newrow[0] = x
							newrow[[x for x in range(1,57)]] = 0.0
							# df = df.append(select)
							if isinstance(newrow, pd.core.series.Series):
								df = pd.concat([df, pd.DataFrame(newrow).T])
							else:
								df = pd.concat([df, newrow])
					select = df.iloc[1].copy()
					for x in ex_res['TEMBANAME'].tolist():
						select[0] = x
						select[[idx for idx in range(1,57)]] = 0.0
						if isinstance(select, pd.core.series.Series):
							df = pd.concat([df, pd.DataFrame(select).T])
						else:
							df = pd.concat([df, select])
					for x in ex_ror['TEMBANAME'].tolist():
						select[0] = x
						select[[idx for idx in range(1,57)]] = 0.0
						if isinstance(select, pd.core.series.Series):
							df = pd.concat([df, pd.DataFrame(select).T])
						else:
							df = pd.concat([df, select])
					for x in fut['TEMBANAME'].tolist():
						select[0] = x
						select[[idx for idx in range(1,57)]] = 0.0
						# enforcing GERD to be operational in 2023
						if 'Renaissance' in x:
							select[2023-year_start+1] = AHA.loc[AHA['TEMBANAME']==x]['Capacity'].values[0]/1000.0
						if set_hydro_capacity_only is True:
							select[[idx for idx in range(1,57)]] = 0.0
							select[[AHA.loc[AHA['TEMBANAME']==x]['First Year'].values[0] - year_start + 1]] = AHA.loc[AHA['TEMBANAME']==x]['Capacity'].values[0]/1000.0

						if isinstance(select, pd.core.series.Series):
							df = pd.concat([df, pd.DataFrame(select).T])
						else:
							df = pd.concat([df, select])
				elif set_capacity is True:
					with open('../jrc_temba-master/output_data/sorted_TEMBA_'+scenario+capacity_scenario+'.txt') as f:
						file = f.read().split("\n")[:-1]
					data = []
					for el in file:
						data.append(el.split('\t'))
					select = []
					for row in data:
						if row[0]=='NewCapacity':
							select.append(row)
					select = pd.DataFrame(select)
					for tech in select[2]:
						if df.loc[df[0]==tech].empty and 'BACKSTOP' not in tech:
							newrow = df.iloc[1].copy()
							newrow[0] = tech
							newrow.loc[[x for x in range(year_start - 2015 + 1, year_end - 2015 + 1 + 1)]] = select.loc[select[2]==tech,[x for x in range(3,3+year_end - year_start + 1)]].values[0]
							if isinstance(newrow, pd.core.series.Series):
								df = pd.concat([df, pd.DataFrame(newrow).T])
							else:
								df = pd.concat([df, newrow])
						else:
							df.loc[df[0]==tech, [x for x in range(year_start - 2015 + 1, year_end - 2015 + 1 + 1)]] = select.loc[select[2]==tech,[x for x in range(3,3+year_end - year_start + 1)]].values[0]	
			## fix the minimum capacity to be installed for AHA hydropower plants
			if sheet in ['CapacityOfOneTechnologyUnit']:
				select = df.iloc[0].copy()
				for x in fut['TEMBANAME'].tolist():
					select[0] = x
					select[[idx for idx in range(1,57)]] = np.max([(fut.loc[AHA['TEMBANAME']==x]['Capacity'].values[0])/1000.0,0.0])
					newrow = select.copy()
					if isinstance(newrow, pd.core.series.Series):
						df = pd.concat([df, pd.DataFrame(newrow).T])
					else:
						df = pd.concat([df, newrow])
			if sheet in ['AccumulatedAnnualDemand']:
				fuels = pd.read_excel('../SSP/'+dataset+'/AccumulatedAnnualDemandSSP.xlsx', index_col=None, header=None)
				if 'SSP' in scenario:
					fuels = fuels.loc[fuels[1].str.contains(scenario)]
					fuels = fuels.drop(1, axis=1)
					columnlist = fuels.columns.tolist()
					fuels.columns = [x for x in range(len(columnlist))]
					cols = [0]
					[cols.append(x) for x in range(1, 1 + year_end - year_start  + 1)]
					df = df[df.columns[cols]]
					fuels = fuels[[fuels.columns[x] for x in range(0, year_end - year_start +2)]]
					df.columns = [x for x in range(len(df.columns))]
					df.iloc[1:] = fuels.iloc[0:]
			if sheet in ['SpecifiedAnnualDemand']:
				elec = pd.read_excel('../SSP/'+dataset+'/SpecifiedAnnualDemandSSP.xlsx', index_col=None, header=None)
				if 'SSP' in scenario:
					elec = elec.loc[elec[1].str.contains(scenario)]
					elec = elec.drop(1, axis=1)
					columnlist = elec.columns.tolist()
					elec.columns = [x for x in range(len(columnlist))]
					cols = [0]
					[cols.append(x) for x in range(1, 1 + year_end - year_start + 1)]
					df = df[df.columns[cols]]
					elec = elec[[fuels.columns[x] for x in range(0, year_end - year_start + 2)]]
					df.iloc[1:] = elec.iloc[0:]
			if sheet in ['TotalTechnologyAnnualActivityUp']:
				if TLines==False:
					select = df.iloc[0].copy()
					for x in PP_lines_all:
						select[0] = x
						select[[idx for idx in range(1, first_hydro_year - year_start)]] = 999999.9
						select[[idx for idx in range(1 + first_hydro_year - year_start , 57)]] = 0.0
						newrow = select.copy()
						if isinstance(newrow, pd.core.series.Series):
							df = pd.concat([df, pd.DataFrame(newrow).T])
						else:
							df = pd.concat([df, newrow])					
				if 'Emission' in sheet and sheet not in ['AnnualEmissionLimit']:
					df = pd.read_excel(TEMBA15, sheet, index_col=None, header=None)
				if 'EMISSION' == sheet:
					df = pd.read_excel(TEMBA15, sheet, index_col=None, header=None)
			## remove the years that are not considered in the optimization
			if not(sheet.isupper()) and len(df.columns)>2 and sheet not in ['AccumulatedAnnualDemand','SpecifiedAnnualDemand']:
				df = df.drop([idx for idx in range(len(df.columns) - 2070 + year_end, len(df.columns))], axis=1)
				df = df.drop([idx for idx in range(len(df.columns) - 55 + (2070 - year_end) - 1, len(df.columns) - 55 + (2070 - year_end) + year_start - 2015 - 1)], axis=1)
			if not(sheet.isupper()) and len(df.columns)>2 and sheet in ['AccumulatedAnnualDemand','SpecifiedAnnualDemand']:
				df = df.drop([idx for idx in range(len(df.columns) - year_end + year_end, len(df.columns))], axis=1)
				df = df.drop([idx for idx in range(len(df.columns) - 30 + (year_end - year_end) - 1, len(df.columns) - 30 + (year_end - year_end) + year_start - 2020 - 1)], axis=1)
			df.to_excel(writer, sheet_name=sheet, header=False, index=False)

		writer.save()

year_start = 2015
year_end = 2050
dataset = 'ISIMIP2b'

hydro_scenario = ['normal','dry','dry','normal','dry','dry','normal','dry']
set_capacity = [False,False,True,True,True,True, False,False]
capacity_scenario = ['','','','_dry','','_dry', '','']
set_hydro_capacity_only = [False,False,False,False,False,False, False,False]
TLines_scenario = [True,True,True,True,False,False, True,True]
hydrology=[None, None, None, None, None, None, 'control','control']
for el in [0,1,6,7]:
# for el in [2,3,4,5]:
	prepare_scenarios(year_start, year_end, dataset, set_capacity[el], 
		capacity_scenario[el], set_hydro_capacity_only[el], hydro_scenario[el],
		TLines=TLines_scenario[el], hydrology=hydrology[el])

