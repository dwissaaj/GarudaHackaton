import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
scaler = MinMaxScaler()
df = pd.read_excel("data.xlsx")
df['province'] = df['Provinsi'].replace({'ACEH':0,'BALI':1,'BANTEN':2,'BENGKULU':3,'DAERAH ISTIMEWA YOGYAKARTA':4,
 'DKI JAKARTA':5,'GORONTALO':6,'JAMBI':7,'JAWA BARAT':8,'JAWA TENGAH':9,'JAWA TIMUR':10,
 'KALIMANTAN BARAT':11,'KALIMANTAN SELATAN':12,'KALIMANTAN TENGAH':13,'KALIMANTAN TIMUR':14,'KALIMANTAN UTARA':15,'KEPULAUAN BANGKA BELITUNG':16
, 'KEPULAUAN RIAU':17,'LAMPUNG':18,'MALUKU':19,'MALUKU UTARA':20,'NUSA TENGGARA BARAT':21,
 'NUSA TENGGARA TIMUR':22,'PAPUA':23,'PAPUA BARAT':24,'RIAU':25,'SULAWESI BARAT':26,
 'SULAWESI SELATAN':27,'SULAWESI TENGAH':28,'SULAWESI TENGGARA':29,'SULAWESI UTARA':30,
 'SUMATERA BARAT':31,'SUMATERA SELATAN':32,'SUMATERA UTARA':33},regex=True)
df = df.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
df2 = pd.get_dummies(data=df,columns=['Jenis_Ikan','peak_month_start','peak_month_end'])
cols_scale = ['Volume_Produksi(TON)','Nilai_Produksi(Rupiah)','height','width',]

df2[cols_scale] = scaler.fit_transform(df2[cols_scale])
X = df2.drop(['Unnamed: 8','Tahun','Provinsi'], axis='columns')

y = df['province']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = SVC()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))

