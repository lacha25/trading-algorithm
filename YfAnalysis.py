import yfinance as yf 

def f_yf_analysis(ent):
    number_of_elements=2
    dat=yf.Ticker(ent)


    priceTarget=dat.analyst_price_targets
    signal=0
    max=priceTarget['high']
    low=priceTarget['low']
    mean=priceTarget['mean']
    current=priceTarget['current']

    if current>mean:
        signal-= min((current - mean) / (max - mean),1)
    elif current<mean:
        signal+=min((mean - current) / (mean - low),1)

    print(f"Price Target for {ent}: {priceTarget}")


    recommandations=dat.recommendations
    sb=recommandations['strongBuy'][0]
    buy=recommandations['buy'][0]
    hold=recommandations['hold'][0]
    sell=recommandations['sell'][0]
    ss=recommandations['strongSell'][0]
    if buy>hold and buy+sb>sell+ss:
        signal=1
    elif sell>hold and sell+ss>buy+sb:
        signal=-1


    return signal/number_of_elements