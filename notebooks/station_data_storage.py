import pandas as pd
import re
import io

# Raw text data provided by the user
station_info_text = """
AAA Alma Ata Kazakhstan 43.25 76.92 closed Edi
AAE Addis Ababa Ethiopia 9.035 38.77 closed Par
ABG Alibag India 18.638 72.872 imo Kyo
ABK Abisko Sweden 68.358 18.823 imo Edi
AIA Faraday Islands (Argentine Island) - Akademik Vernadsky base Antarctica -65.245 295.742 imo Edi
ALE Alert Canada 82.497 297.647 closed Ott
AMS Martin de Vivies - Amsterdam Island French Southern and Antarctic Lands -37.8 77.57 closed Par
API Apia Western Samoa -13.8155 -171.7812 imo Edi
AQU L'Aquila Italy 42.38 13.32 closed Par
ARS Arti Russia 56.433 58.567 imo Edi
ASC Ascension Island United Kingdom -7.95 345.62 imo Edi
ASP Alice Springs Australia -23.762 133.883 imo Edi
BDV Budkov Czech Republic 49.08 14.02 imo Edi
BEL Belsk Poland 51.836 20.789 imo Edi
BFE Brorfelde Denmark 55.625 11.672 closed Kyo
BFO Black Forest Germany 48.331 8.325 imo Edi
BLC Baker Lake Canada 64.318 263.988 imo Ott
BMT Beijing Ming Tombs China 40.3 116.2 imo Kyo
BNG Bangui Central African Republic 4.33 18.57 closed Par
BOU Boulder United States of America 40.14 254.767 imo Gol
BOX Borok Russia 58.07 38.23 imo Par
BRD Brandon Canada 49.87 260.0261 imo Ott
BRW Barrow United States of America 71.32 203.38 imo Gol
BSL Stennis Space Center (Bay St. Louis) United States of America 30.35 270.36 imo Gol
CBB Cambridge Bay Canada 69.123 254.969 imo Ott
CKI Cocos (Keeling) Islands Australia -12.1875 96.8336 imo Edi
CLF Chambon-la-Foret France 48.025 2.26 imo Par
CMO College United States of America 64.87 212.14 imo Gol
CNB Canberra Australia -35.32 149.36 imo Edi
CNH Changchun China 44.08 124.86 closed Edi
CPL Choutuppal 17.294 78.9192 imo Edi
CSY Casey Station Antarctica -66.283 110.533 imo Edi
CTA Charters Towers Australia -20.09 146.264 imo Edi
CYG Cheongyang Korea 36.37 126.854 imo Edi
CZT Port Alfred French Southern and Antarctic Lands -46.43 51.86 closed Par
DED Deadhorse United States of America 70.36 211.21 imo Gol
DLR Del Rio United States of America 29.49 259.083 closed Gol
DLT Dalat Vietnam 11.95 108.48 imo Par
DMC Dome Concordia Antarctica -75.25 124.167 closed Par
DOU Dourbes Belgium 50.1 4.6 imo Edi
DRV Dumont d'Urville Antarctica -66.67 140.01 closed Par
DUR Duronia Italy 41.65 14.47 imo Par
EBR Ebro Spain 40.957 0.333 imo Par
ESK Eskdalemuir United Kingdom 55.314 356.794 imo Edi
EYR Eyrewell New Zealand -43.474 172.393 imo Edi
FCC Fort Churchill Canada 58.759 265.912 imo Ott
FRD Fredericksburg United States of America 38.21 282.633 imo Gol
FRN Fresno United States of America 37.09 240.28 imo Gol
FUR Furstenfeldbruck Germany 48.17 11.28 imo Edi
GAN Gan International Airport Maldives -0.6946 73.15374 imo Edi
GCK Grocka Republic of Serbia 44.633 20.767 imo Edi
GDH Qeqertarsuaq (Godhavn) Greenland 69.252 306.467 imo Kyo
GLN Glenlea Canada 49.645 262.88 closed Ott
GNA Gnangara Australia -31.78 115.947 closed Edi
GNG Gingin Australia -31.356 115.715 imo Edi
GUA Guam United States of America 13.59 144.87 imo Gol
GUI Guimar Spain 28.321 343.559 imo Par
GZH Zhaoqing China 23.97 112.45 closed Edi
HAD Hartland United Kingdom 51 355.52 imo Edi
HBK Hartebeesthoek South Africa -25.88 27.71 imo Edi
HER Hermanus South Africa -34.43 19.23 imo Edi
HLP Hel Poland 54.6035 18.8107 imo Edi
HON Honolulu United States of America 21.32 202 imo Gol
HRB Hurbanovo Slovakia 47.873 18.19 imo Par
HRN Hornsund Norway 77 15.55 imo Edi
HUA Huancayo Peru -12.05 284.67 imo Edi
HYB Hyderabad India 17.42 78.55 imo Edi
IPM Isla de Pascua Mataveri (Easter Island) Chile -27.1713 250.58 closed Par
IQA Iqaluit Canada 63.753 291.482 imo Ott
IRT Irkutsk (Patrony) Russia 52.17 104.45 imo Edi
ISK Istanbul-Kandilli Turkey 41.063 29.062 closed Edi
IZN Iznik Turkey 40.5 29.72 imo Edi
JAI Jaipur India 26.92 75.8 imo Kyo
JCO Jim Carrigan Observatory United States of America 70.356 -148.799 imo Edi
KAK Kakioka Japan 36.232 140.186 imo Kyo
KDU Kakadu Australia -12.69 132.47 imo Edi
KEP King Edward Point South Georgia and the South Sandwich Islands -54.2821 323.5071 imo Edi
KHB Khabarovsk Russia 47.61 134.69 imo Edi
KIR Kiruna 67.8428 20.4201 imo Edi
KIV Kiev Ukraine 50.72 30.3 imo Edi
KMH Keetmanshoop Namibia -26.54 18.11 imo Edi
KNY Kanoya Japan 31.42 130.88 imo Kyo
KOU Kourou French Guiana 5.21 307.27 imo Par
LER Lerwick United Kingdom 60.138 358.817 imo Edi
LNP Lunping Taiwan 25 121.167 closed Kyo
LON Lonjsko Polje Croatia 45.4081 16.6592 imo Edi
LOV Lovo Sweden 59.34 17.82 closed Edi
LRM Learmonth Australia -22.22 114.1 imo Edi
LVV Lviv Ukraine 49.9 23.75 imo Edi
LYC Lycksele Sweden 64.612 18.748 imo Edi
LZH Lanzhou China 36.087 103.845 closed Par
MAB Manhay Belgium 50.298 5.682 imo Edi
MAW Mawson Antarctica -67.6 62.88 imo Edi
MBC Mould Bay Canada 76.315 240.638 closed Ott
MBO Mbour Senegal 14.39 343.04 closed Par
MCQ Macquarie Island Australia -54.5 158.95 imo Edi
MEA Meanook Canada 54.616 246.653 imo Ott
MGD Magadan Russia 60.051 150.728 imo Edi
MID Midway Island United States of America 28.21 182.62 closed Gol
MLT Misallat 29.515 30.892 closed Edi
MMB Memambetsu Japan 43.91 144.19 imo Kyo
NAQ Narsarsuaq Greenland 61.167 314.567 imo Kyo
NCK Nagycenk Hungary 47.63 16.72 imo Edi
NEW Newport United States of America 48.27 242.88 imo Gol
NGK Niemegk Germany 52.07 12.68 imo Edi
NUR Nurmijarvi Finland 60.51 24.66 imo Edi
NVS Novosibirsk (Klyuchi) Russia 54.85 83.23 imo Edi
ORC Orcadas Argentina -60.737 -44.737 imo Edi
OTT Ottawa Canada 45.403 284.448 imo Ott
PAF Port-aux-Francais French Southern and Antarctic Lands -49.35 70.26 closed Par
PAG Panagjurishte Bulgaria 42.515 24.177 imo Edi
PBQ Poste-de-la-Baleine Canada 55.277 282.255 closed Ott
PEG Pedeli Greece 38.1 23.9 imo Edi
PET Paratunka (Petropavlovsk) Russia 52.971 158.248 imo Edi
PHU Phuthuy Vietnam 21.03 105.96 imo Par
PIL Pilar Argentina -31.667 -63.881 imo Edi
PPT Pamatai (Papeete) French Polynesia -17.567 210.426 imo Par
PST Port Stanley Falkland Islands (Islas Malvinas) -51.7 302.11 imo Edi
QSB Qsaybeh Lebanon 33.871 35.644 closed Par
RES Resolute Bay Canada 74.69 265.105 imo Ott
SBA Scott Base Antarctica -77.829 166.671 imo Edi
SBL Sable Island Canada 43.9321 299.9905 imo Edi
SFS San Fernando Spain 36.667 354.055 imo Par
SHE Saint Helena Saint Helena, Ascension and Tristan da Cunha, British Overseas Territories -15.961 354.253 imo Kyo
SHU Shumagin United States of America 55.35 199.54 imo Gol
SIT Sitka United States of America 57.06 224.67 imo Gol
SJG San Juan United States of America 18.11 293.85 imo Gol
SOD Sodankyla Finland 67.37 26.63 imo Edi
SON Sonmiani Pakistan 25.1168 66.4487 closed Edi
SPG Saint Petersburg Russia 60.542 29.716 imo Par
SPT San Pablo-Toledo Spain 39.55 -4.35 imo Par
STJ St John's Canada 47.595 307.323 imo Ott
STT San Teotonio 37.5467 -8.7277 imo Kyo
SUA Surlari Romania 44.68 26.25 imo Par
TAM Tamanrasset Algeria 22.79 5.53 imo Par
TAN Antananarivo Madagascar -18.917 47.552 closed Par
TDC Tristan da Cunha Saint Helena, Ascension and Tristan da Cunha, British Overseas Territories -37.067 -12.316 imo Kyo
TEO Teoloyucan Mexico 19.747 260.818 closed Edi
THL Qaanaaq (Thule) Greenland 77.47 290.773 imo Kyo
THY Tihany Hungary 46.9 17.89 imo Edi
TIK Tiksi Russia 71.58 129 closed Par
TRW Trelew Argentina -43.267 294.617 closed Edi
TSU Tsumeb Namibia -19.202 17.584 imo Edi
TTB Tatuoca -1.205 311.487 imo Edi
TUC Tucson United States of America 32.17 249.27 imo Gol
UPS Uppsala (Fiby) Sweden 59.903 17.353 imo Edi
VAL Valentia Republic of Ireland 51.933 349.75 imo Edi
VIC Victoria Canada 48.52 236.58 imo Ott
VNA Neumayer Station III Antarctica -70.683 -8.282 imo Kyo
VOS Vostok Antarctica -78.464 106.835 imo Kyo
VSS Vassouras Brazil -22.4 316.35 imo Par
WIC Conrad Observatory Austria 47.9305 15.8657 imo Edi
WMQ Urumqi China 43.81 87.71 closed Edi
WNG Wingst Germany 53.725 9.053 imo Edi
YAK Yakutsk Russia 61.96 129.66 imo Edi
YKC Yellowknife Canada 62.48 245.518 imo Ott
"""

code_list_text = """
cta hrb mcq sjg
aaa ctry_inf hrn mea sod
aae czt hua mmb spt
abg ded hyb naq stj
abk dou ipm nck sua
aia drv iqa new tam
api ebr irt ngk tdc
asc esk izn nur thl
asp eyr jai nvs thy
bdv fcc kak obsy_inf trw
bel frd kdu ott tsu
bfo frn kiv paf tuc
blc fur kmh pag ups
bmt gck kny pet val
bou gdh kou phu vic
box gna ler ppt vos
brw gua lrm pst vss
bsl gui lvv res wng
cbb had lyc sba yak
clf hbk lzh sfs ykc
cmo her mab she
cnb hlp maw shu
csy hon mbo sit
"""

# --- Parsing Logic ---
all_station_data = {}
for line in io.StringIO(station_info_text).readlines():
    if not line.strip():
        continue
    parts = line.split()
    code = parts[0]
    # Find the float values for lat/lon
    coords = re.findall(r"-?\d+\.\d+", line)
    if len(coords) >= 2:
        lat, lon = float(coords[0]), float(coords[1])
        all_station_data[code] = {"GEOLAT": lat, "GEOLON": lon}

# Get the list of desired codes and clean it up
desired_codes = set(code.upper() for code in code_list_text.split())
undesired = {"CTRY_INF", "OBSY_INF", "?"}
desired_codes = desired_codes - undesired

# Build the final list for the DataFrame
output_data = []
for code in sorted(list(desired_codes)):
    if code in all_station_data:
        output_data.append(
            {
                "IAGA": code,
                "GEOLAT": all_station_data[code]["GEOLAT"],
                "GEOLON": all_station_data[code]["GEOLON"],
            }
        )

# Create and save the CSV
df_stations = pd.DataFrame(output_data)
output_filename = "stations_full_list.csv"
df_stations.to_csv(output_filename, index=False)

print(f"Successfully created '{output_filename}' with {len(df_stations)} stations.")
