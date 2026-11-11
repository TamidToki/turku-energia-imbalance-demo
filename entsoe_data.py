"""Download Finland day-ahead prices from the authorised ENTSO-E API.
Token is read from ENTSOE_TOKEN_FILE, never logged or stored in the repo.
"""
import os,time,json,calendar
from pathlib import Path
import xml.etree.ElementTree as ET
import requests
import pandas as pd

API='https://web-api.tp.entsoe.eu/api'
AREA='10YFI-1--------U'

def parse_prices(xml):
    root=ET.fromstring(xml)
    def tag(e): return e.tag.rsplit('}',1)[-1]
    def child(e,name): return next((x for x in e if tag(x)==name),None)
    rows=[]
    for ts in root.iter():
        if tag(ts)!='TimeSeries': continue
        for period in ts:
            if tag(period)!='Period': continue
            interval=child(period,'timeInterval')
            start=pd.Timestamp(child(interval,'start').text)
            end=pd.Timestamp(child(interval,'end').text)
            resolution=child(period,'resolution').text
            delta=pd.Timedelta(resolution)
            points=[]
            for p in period:
                if tag(p)=='Point':
                    points.append((int(child(p,'position').text),float(child(p,'price.amount').text)))
            points.sort()
            # ENTSO-E A03 curves encode changes only: fill until the next point.
            for i,(pos,value) in enumerate(points):
                a=start+(pos-1)*delta
                b=start+(points[i+1][0]-1)*delta if i+1<len(points) else end
                for stamp in pd.date_range(a,b,freq=delta,inclusive='left'):
                    rows.append((stamp,value))
    if not rows:
        reasons=[e.text for e in root.iter() if tag(e)=='text']
        raise RuntimeError('No prices returned: '+str(reasons))
    return pd.DataFrame(rows,columns=['timestamp_utc','value'])

def main():
    key=Path(os.environ['ENTSOE_TOKEN_FILE']).read_text().strip()
    out=Path(__file__).parent/'data'/'entsoe';out.mkdir(exist_ok=True,parents=True)
    frames=[]
    for month in range(5,12):
        a=pd.Timestamp(year=2026,month=month,day=1,tz='UTC');b=a+pd.offsets.MonthBegin(1)
        cache=out/f'fi_2026_{month:02}.xml'
        if not cache.exists():
            r=requests.get(API,params=dict(securityToken=key,documentType='A44',in_Domain=AREA,out_Domain=AREA,periodStart=a.strftime('%Y%m%d%H%M'),periodEnd=b.strftime('%Y%m%d%H%M')),timeout=90)
            if r.status_code!=200: raise RuntimeError(f'ENTSO-E HTTP {r.status_code}, credentials withheld')
            parse_prices(r.text)
            cache.write_text(r.text)
            time.sleep(1)
        df=parse_prices(cache.read_text());frames.append(df)
        print(cache.name,len(df),flush=True)
    df=pd.concat(frames).drop_duplicates('timestamp_utc').sort_values('timestamp_utc')
    df.to_csv(out/'spot_fi.csv',index=False)
    print('Saved',len(df),'price intervals to data/entsoe/spot_fi.csv')
if __name__=='__main__':main()
