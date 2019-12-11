import pandas as pd

d = pd.read_excel('./triple_threat.xlsx')

d = d.drop(columns='action')

def all_condition(data):
    if data.shape[1] == 1:
        return data.dropna()
    
    first, rest = data[data.columns[0]], data[data.columns[1:]]
    first = first.dropna()
    
    rest_condition = all_condition(rest)
    res = pd.DataFrame()
    
    for v in first:
        c = rest_condition.copy()
        c.insert(loc=0, column=first.name, value=[v] * c.shape[0])
        res = pd.concat([res, c])

    return res

r = all_condition(d)

r.to_excel('triple_all.xlsx', index=False)

