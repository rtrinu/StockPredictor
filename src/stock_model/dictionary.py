import pandas as pd
import os

def fetch_stock_data() -> dict:
    """
    Fetches the S&P 500 companies' information and returns a dictionary mapping
    stock symbols to company names.

    :return: dict - Dictionary with stock symbols as keys and company names as values.
    """
    if not os.path.exists('sp500_FTSE100_stocks.csv'):
        url = 'https://datahub.io/core/s-and-p-500-companies/r/constituents.csv'
        sp500 = pd.read_csv(url)
        sp500.to_csv('sp500_FTSE100_stocks.csv', index=False)
        getFTSE100_data()

    else:
        sp500 = pd.read_csv('sp500_FTSE100_stocks.csv')
        
    return dict(zip(sp500['Symbol'], sp500['Security']))

def getFTSE100_data():
    ftse100_dict = {
    'AAL.L': 'Anglo American plc',
    'ADBE.L': 'Adobe Inc.',
    'ADM.L': 'ADM (Archer-Daniels-Midland Company)',
    'AFL.L': 'Aflac Inc.',
    'AGK.L': 'Aggreko plc',
    'ALC.L': 'Alcon Inc.',
    'AMG.L': 'Amgen Inc.',
    'AMT.L': 'American Tower Corporation',
    'AV.L': 'Aviva plc',
    'BA.L': 'British Airways (International Airlines Group)',
    'BARC.L': 'Barclays plc',
    'BATS.L': 'British American Tobacco plc',
    'BHP.L': 'BHP Group plc',
    'BP.L': 'BP plc',
    'BRBY.L': 'Burberry Group plc',
    'CCH.L': 'Coca-Cola HBC AG',
    'CNA.L': 'Centrica plc',
    'CPG.L': 'Capita plc',
    'CRH.L': 'CRH plc',
    'DGE.L': 'Diageo plc',
    'DLG.L': 'Direct Line Insurance Group plc',
    'DNO.L': 'DNO ASA',
    'ECO.L': 'Ecopetrol S.A.',
    'ENB.L': 'Enbridge Inc.',
    'ENRC.L': 'Eurasian Natural Resources Corporation',
    'EVR.L': 'Evraz plc',
    'GSK.L': 'GlaxoSmithKline plc',
    'HL.L': 'Hargreaves Lansdown plc',
    'HSBA.L': 'HSBC Holdings plc',
    'IAG.L': 'International Airlines Group',
    'IMB.L': 'Imperial Brands plc',
    'INF.L': 'Informa plc',
    'ITV.L': 'ITV plc',
    'JD.L': 'JD Sports Fashion plc',
    'JMAT.L': 'Johnson Matthey plc',
    'LGEN.L': 'Legal & General Group plc',
    'LLOY.L': 'Lloyds Banking Group plc',
    'MNG.L': 'M&G plc',
    'NG.L': 'National Grid plc',
    'NXT.L': 'Next plc',
    'OML.L': 'Old Mutual Limited',
    'PHNX.L': 'Phoenix Group Holdings plc',
    'PRU.L': 'Prudential plc',
    'RB.L': 'Reckitt Benckiser Group plc',
    'REL.L': 'Relx plc',
    'RIO.L': 'Rio Tinto Group',
    'RMG.L': 'Royal Mail plc',
    'RRS.L': 'Randgold Resources Limited',
    'RDSA.L': 'Royal Dutch Shell plc',
    'SAB.L': 'SABMiller plc',
    'SBRY.L': 'Sainsbury\'s plc',
    'SDR.L': 'Schroders plc',
    'SGE.L': 'SEGRO plc',
    'SHEL.L': 'Shell plc',
    'SMT.L': 'Scottish Mortgage Investment Trust plc',
    'SN.L': 'Smith & Nephew plc',
    'SPG.L': 'Segro plc',
    'SSE.L': 'SSE plc',
    'STAN.L': 'Standard Chartered plc',
    'TSCO.L': 'Tesco plc',
    'ULVR.L': 'Unilever plc',
    'VOD.L': 'Vodafone Group plc',
    'WPP.L': 'WPP plc',
    'META':'Meta Platforms'
    }   
    df = pd.read_csv('sp500_FTSE100_stocks.csv')
    new_data = pd.DataFrame(list(ftse100_dict.items()),columns=['Symbol','Security'])
    updated_df = pd.concat([df,new_data],ignore_index=True)
    df = updated_df.sort_values(by='Symbol')
    df.to_csv('sp500_FTSE100_stocks.csv', index=False)   
     

    


def get_stock_symbol_from_name(user_stock: str, stock_dict: dict) -> tuple:
    """
    Fetches the stock symbol from the stock dictionary using either the stock symbol or company name.

    :param user_stock: str, stock symbol or company name.
    :param stock_dict: dict, dictionary of stock symbols and company names.
    :return: tuple - Stock symbol and stock name (or None if not found).
    """
    if user_stock in stock_dict:
        return user_stock, stock_dict[user_stock]
    else:
        for symbol, name in stock_dict.items():
            if user_stock.lower() in name.lower():
                return symbol, name
    return None, None