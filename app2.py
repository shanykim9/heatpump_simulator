"""
=============================================================================
[스마트 히트펌프 에너지 및 요금 시뮬레이터 : R290 vs R32 듀얼 비교 엔진]

1. 열역학 및 듀얼 물리 모델 
   - [근본적 수정] R290 및 R32 카탈로그 원본 Data 100% 독립 하드코딩 이식 완료.
   - 스마트 제상 로직: R290은 수식 감쇠 적용, R32는 카탈로그 반영으로 패스.
   - 극한 온도 백업 히터: 운전 불가 영역(-) 진입 시 정격 용량(kW) 전기히터 가동 옵션.

2. 제어 및 운영 알고리즘
   - 스마트 제어: 개편 요금제에 맞춘 주말 낮 축열 및 저녁 피크 회피 로직.
   - 평일/주말 온수 스케줄 분리: 법정 공휴일 자동 인식 및 패턴 전환.

3. 최신 전기요금 산출 (2026년 전기요금 개편안 완벽 반영)
   - 일반용 전력: 경부하 단가 인상, 최대부하(피크) 단가 인하 및 18~21시 변경.
   - 주말 파격 할인: 봄/가을 주말 및 공휴일 11~14시 일반용 50% 할인.
=============================================================================
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import RegularGridInterpolator
from geopy.geocoders import Nominatim
import requests
import urllib.parse
from datetime import datetime, timedelta
import pandas as pd
import holidays
import xml.etree.ElementTree as ET

# ==========================================
# 1. 행정구역 및 지역난방(KDHC) 기준 데이터
# ==========================================
KOREA_LOCATIONS = {
    "서울특별시": ["강남구", "강동구", "강북구", "강서구", "관악구", "광진구", "구로구", "금천구", "노원구", "도봉구", "동대문구", "동작구", "마포구", "서대문구", "서초구", "성동구", "성북구", "송파구", "양천구", "영등포구", "용산구", "은평구", "종로구", "중구", "중랑구"],
    "부산광역시": ["강서구", "금정구", "기장군", "남구", "동구", "동래구", "부산진구", "북구", "사상구", "사하구", "서구", "수영구", "연제구", "영도구", "중구", "해운대구"],
    "대구광역시": ["군위군", "남구", "달서구", "달성군", "동구", "북구", "서구", "수성구", "중구"],
    "인천광역시": ["강화군", "계양구", "남동구", "동구", "미추홀구", "부평구", "서구", "연수구", "옹진군", "중구"],
    "광주광역시": ["광산구", "남구", "동구", "북구", "서구"],
    "대전광역시": ["대덕구", "동구", "서구", "유성구", "중구"],
    "울산광역시": ["남구", "동구", "북구", "울주군", "중구"],
    "세종특별자치시": ["세종시"],
    "경기도": ["가평군", "고양시", "과천시", "광명시", "광주시", "구리시", "군포시", "김포시", "남양주시", "동두천시", "부천시", "성남시", "수원시", "시흥시", "안산시", "안성시", "안양시", "양주시", "양평군", "여주시", "연천군", "오산시", "용인시", "의왕시", "의정부시", "이천시", "파주시", "평택시", "포천시", "하남시", "화성시"],
    "강원특별자치도": ["강릉시", "고성군", "동해시", "삼척시", "속초시", "양구군", "양양군", "영월군", "원주시", "인제군", "정선군", "철원군", "춘천시", "태백시", "평창군", "홍천군", "화천군", "횡성군"],
    "충청북도": ["괴산군", "단양군", "보은군", "영동군", "옥천군", "음성군", "제천시", "증평군", "진천군", "청주시", "충주시"],
    "충청남도": ["계룡시", "공주시", "금산군", "논산시", "당진시", "보령시", "부여군", "서산시", "서천군", "아산시", "예산군", "천안시", "청양군", "태안군", "홍성군"],
    "전북특별자치도": ["고창군", "군산시", "김제시", "남원시", "무주군", "부안군", "순창군", "완주군", "익산시", "임실군", "장수군", "전주시", "정읍시", "진안군"],
    "전라남도": ["강진군", "고흥군", "곡성군", "광양시", "구례군", "나주시", "담양군", "목포시", "무안군", "보성군", "순천시", "신안군", "여수시", "영광군", "영암군", "완도군", "장성군", "장흥군", "진도군", "함평군", "해남군", "화순군"],
    "경상북도": ["경산시", "경주시", "고령군", "구미시", "김천시", "문경시", "봉화군", "상주시", "성주군", "안동시", "영덕군", "영양군", "영주시", "영천시", "예천군", "울릉군", "울진군", "의성군", "청도군", "청송군", "칠곡군", "포항시"],
    "경상남도": ["거제시", "거창군", "고성군", "김해시", "남해군", "밀양시", "사천시", "산청군", "양산시", "의령군", "진주시", "창녕군", "창원시", "통영시", "하동군", "함안군", "함양군", "합천군"],
    "제주특별자치도": ["서귀포시", "제주시"]
}

def get_climate_region(sido, gungu):
    if "제주" in sido: return 0.7
    if sido in ["부산광역시", "대구광역시", "울산광역시", "광주광역시", "전라남도", "경상남도"]: return 0.85
    cold_gungu = ["연천군", "포천시", "가평군", "양평군", "파주시", "동두천시", "양주시", "의정부시", 
                  "철원군", "화천군", "양구군", "인제군", "고성군", "태백시", "평창군", "정선군", "홍천군",
                  "제천시", "단양군", "음성군", "괴산군"]
    if "강원" in sido or (gungu in cold_gungu): return 1.1
    return 1.0

def get_kdhc_region(sido, gungu):
    if "세종" in sido or gungu == "파주시": return 'A'
    if gungu in ["고양시", "수원시", "용인시", "평택시", "청주시"]: return 'B'
    if "서울" in sido or gungu in ["성남시", "화성시"] or "대구" in sido or "광주" in sido: return 'C'
    if gungu in ["김해시", "양산시"]: return 'D'
    northern_gyeonggi = ["동두천시", "연천군", "가평군", "의정부시", "양주시", "포천시"]
    if "강원" in sido or "충청남도" in sido or "대전" in sido or (sido == "경기도" and gungu in northern_gyeonggi): return 'A'
    if "인천" in sido or "경기" in sido or "충청북도" in sido: return 'B'
    if "경상북도" in sido or "전북" in sido: return 'C'
    if "부산" in sido or "울산" in sido or "경상남도" in sido or "전라남도" in sido or "제주" in sido: return 'D'
    return 'C'

def get_unit_heat_load(region, year_opt, area, type_opt):
    if type_opt == 'house': return 63.0 * 1.163
    year_group = '2009' if year_opt in [1.4, 0.7] else '2025'
    if year_group == '2009':
        if region == 'A': return 55.0 if area > 60 else (57.7 if area > 45 else (58.8 if area > 33 else 60.5))
        elif region == 'B': return 53.0 if area > 60 else (55.6 if area > 45 else (56.7 if area > 33 else 58.3))
        elif region == 'C': return 49.0 if area > 60 else (51.4 if area > 45 else (52.4 if area > 33 else 53.9))
        elif region == 'D': return 45.0 if area > 60 else (47.2 if area > 45 else (48.1 if area > 33 else 49.5))
    else: 
        if region == 'A': return 48.7 if area > 85 else (50.0 if area >= 60 else 52.0)
        elif region == 'B': return 44.9 if area > 85 else (45.8 if area >= 60 else 47.7)
        elif region == 'C': return 41.0 if area > 85 else (42.1 if area >= 60 else 43.8)
        elif region == 'D': return 38.3 if area > 85 else (39.2 if area >= 60 else 41.0)
    return 50.0

# ==========================================
# 2. 히트펌프 다중 냉매 성능 데이터베이스 (TRUE RAW DATA)
# ==========================================
WATER_TEMPS = np.array([35, 40, 45, 50, 55, 60, 65, 70, 75])
AMB_TEMPS = np.array([-25, -20, -15, -10, -7, -5, 2, 7, 10, 15, 20, 25, 30, 35, 43])

# [근본 수정] R290 (12kW U30) 원본 데이터 하드코딩
TRUE_TC_R290 = [
    [4.90, 5.70, 6.55, 7.75, 8.15, 8.45, 9.55, 12.30, 13.00, 13.90, 15.01, 15.00, 16.19, 16.01, 17.02],
    [4.68, 5.54, 6.38, 7.54, 8.05, 8.34, 9.34, 11.99, 12.58, 13.54, 14.65, 14.60, 15.78, 15.43, 16.43],
    [4.50, 5.40, 6.28, 7.40, 7.90, 8.23, 9.20, 11.80, 12.50, 13.40, 14.40, 13.98, 14.63, 14.63, 15.64],
    [4.30, 5.25, 6.12, 7.20, 7.80, 8.12, 9.00, 11.50, 12.10, 13.05, 14.05, 13.62, 14.26, 14.10, 15.10],
    [4.05, 5.00, 5.90, 7.00, 7.70, 8.00, 8.75, 11.20, 11.80, 12.70, 13.60, 13.16, 13.80, 12.60, 13.70],
    [3.80, 4.75, 5.65, 6.78, 7.50, 7.80, 8.40, 10.70, 11.25, 12.10, 12.90, 12.70, 13.43, 11.40, 12.30],
    [0.00, 0.00, 5.40, 6.55, 7.18, 7.60, 8.10, 10.00, 10.50, 10.77, 11.53, 12.30, 13.25, 10.00, 11.06],
    [0.00, 0.00, 0.00, 5.90, 6.45, 6.71, 7.20, 8.75, 9.20, 9.80, 10.60, 11.45, 12.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 4.95, 5.28, 6.10, 7.55, 7.95, 8.30, 9.00, 9.00, 0.00, 0.00, 0.00]
]

TRUE_COP_R290 = [
    [1.85, 2.05, 2.32, 2.73, 2.84, 2.91, 3.20, 3.97, 4.17, 4.43, 4.75, 5.45, 5.85, 6.81, 7.03],
    [1.72, 1.92, 2.16, 2.50, 2.64, 2.70, 2.92, 3.64, 3.79, 4.03, 4.32, 4.93, 5.28, 6.17, 6.18],
    [1.61, 1.81, 2.03, 2.31, 2.42, 2.46, 2.71, 3.30, 3.46, 3.68, 3.91, 4.33, 4.48, 5.58, 5.78],
    [1.44, 1.69, 1.88, 2.12, 2.25, 2.28, 2.47, 3.03, 3.15, 3.35, 3.56, 3.91, 4.05, 4.52, 4.72],
    [1.28, 1.51, 1.68, 1.94, 2.08, 2.11, 2.26, 2.76, 2.88, 3.01, 3.19, 3.52, 3.65, 4.13, 4.39],
    [1.13, 1.34, 1.49, 1.76, 1.90, 1.93, 2.04, 2.46, 2.55, 2.68, 2.81, 3.16, 3.32, 3.80, 3.98],
    [0.00, 0.00, 1.33, 1.59, 1.70, 1.78, 1.86, 2.17, 2.23, 2.41, 2.52, 2.67, 2.86, 3.41, 3.69],
    [0.00, 0.00, 0.00, 1.42, 1.52, 1.58, 1.66, 1.91, 1.97, 2.21, 2.36, 2.54, 2.70, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 1.09, 1.15, 1.40, 1.64, 1.69, 1.77, 1.80, 1.90, 0.00, 0.00, 0.00]
]

# [근본 수정] R32 (12kW HBW1200A2A) 원본 데이터 하드코딩
TRUE_TC_R32 = [
    [8.50, 10.00, 11.50, 11.75, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00],
    [8.25, 9.88, 11.50, 11.75, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00],
    [8.00, 9.75, 11.50, 11.75, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00],
    [0.00, 9.63, 11.50, 11.75, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00],
    [0.00, 0.00, 11.50, 11.75, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00],
    [0.00, 0.00, 0.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00, 12.00], # 65도 출수: 영하 5도 이하 가동불가
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
]

TRUE_COP_R32 = [
    [1.85, 2.13, 2.40, 2.70, 3.00, 3.17, 3.50, 4.60, 4.87, 5.56, 6.02, 6.70, 7.00, 7.41, 7.00],
    [1.58, 1.91, 2.25, 2.55, 2.85, 2.97, 3.31, 4.27, 4.51, 5.13, 5.56, 6.20, 6.50, 6.84, 6.00],
    [1.30, 1.70, 2.10, 2.40, 2.70, 2.78, 3.12, 3.93, 4.16, 4.71, 5.10, 5.70, 6.00, 6.28, 5.50],
    [0.00, 1.49, 1.95, 2.25, 2.55, 2.59, 2.93, 3.60, 3.81, 4.28, 4.64, 5.10, 5.40, 5.71, 5.00],
    [0.00, 0.00, 1.80, 2.10, 2.40, 2.39, 2.73, 2.80, 3.46, 3.85, 4.17, 4.60, 4.90, 5.14, 4.50],
    [0.00, 0.00, 0.00, 1.80, 2.25, 2.20, 2.54, 2.60, 3.10, 3.43, 3.71, 4.10, 4.30, 4.57, 4.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 2.35, 2.60, 2.75, 3.00, 3.25, 3.50, 3.80, 4.00, 3.80], # 65도 출수: 영하 5도 이하 가동불가
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
]

def generate_hp_matrix(base_tc, base_cop, scale_factor):
    tc = np.array(base_tc) * scale_factor
    cop = np.array(base_cop)
    return {'TC': tc.tolist(), 'COP': cop.tolist()}

HP_DATABASE = {
    'R290': {
        12: generate_hp_matrix(TRUE_TC_R290, TRUE_COP_R290, 1.0),
        14: generate_hp_matrix(TRUE_TC_R290, TRUE_COP_R290, 14.0/12.0),
        22: generate_hp_matrix(TRUE_TC_R290, TRUE_COP_R290, 22.0/12.0),
    },
    'R32': {
        9: generate_hp_matrix(TRUE_TC_R32, TRUE_COP_R32, 9.0/12.0),
        12: generate_hp_matrix(TRUE_TC_R32, TRUE_COP_R32, 1.0),
        16: generate_hp_matrix(TRUE_TC_R32, TRUE_COP_R32, 16.0/12.0),
    }
}

class RealHeatPumpModel:
    def __init__(self, refrigerant, nominal_kw):
        self.refrigerant = refrigerant
        self.nominal_kw = nominal_kw
        data = HP_DATABASE[refrigerant][nominal_kw]
        self.interp_cap = RegularGridInterpolator((WATER_TEMPS, AMB_TEMPS), data['TC'], bounds_error=False, fill_value=None)
        self.interp_cop = RegularGridInterpolator((WATER_TEMPS, AMB_TEMPS), data['COP'], bounds_error=False, fill_value=None)

    def get_performance(self, t_amb, t_water):
        cap = self.interp_cap((min(t_water, 75.0), t_amb))
        cop = self.interp_cop((min(t_water, 75.0), t_amb))
        
        # 0.1 이하면 운전 불가 영역 (백업히터 또는 정지 처리)
        if cap is None or cap <= 0.1 or cop <= 0.1:
            return 0.0, 0.0
            
        # R290은 자연 제상 감쇠 적용, R32는 표에 이미 깎여서 반영되어 있으므로 패스
        if self.refrigerant == 'R290':
            if t_amb <= -25: defrost = 0.8
            elif t_amb >= 5: defrost = 1.0
            else: defrost = 0.8 + (t_amb + 25) * (0.2 / 30)
        else:
            defrost = 1.0
            
        return float(cap * 1000 * defrost), float(cop)

# ==========================================
# 3. 외부 API 관리 (Weather, NASA POWER, SMP/REC)
# ==========================================
class DataManager:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="hp_sim_korea_v14")
        self.public_api_key = urllib.parse.unquote("1af73ef98b58a6312cf4ccc024f433b30d4aadc5e0441a5f1258984507693fed")

    def get_coordinates(self, address):
        try:
            loc = self.geolocator.geocode(f"South Korea, {address}")
            return (loc.latitude, loc.longitude) if loc else (None, None)
        except: return None, None

    def get_weather_data(self, lat, lon, start_date, end_date):
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {"latitude": lat, "longitude": lon, "start_date": start_date, "end_date": end_date, "hourly": "temperature_2m", "timezone": "auto"}
        try:
            res = requests.get(url, params=params).json()
            if "hourly" not in res: return None, "기상 데이터 수신 실패"
            hourly_temps = np.array(res["hourly"]["temperature_2m"])
            x_hours = np.arange(len(hourly_temps)) * 60
            x_minutes = np.arange(len(hourly_temps) * 60)
            return np.interp(x_minutes, x_hours, hourly_temps), None
        except Exception as e: return None, str(e)

    def get_solar_irradiance(self, lat, lon, start_date, end_date):
        start_str = start_date.replace("-", "")
        end_str = end_date.replace("-", "")
        url = f"https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=ALLSKY_SFC_SW_DWN&community=RE&longitude={lon}&latitude={lat}&start={start_str}&end={end_str}&format=JSON"
        try:
            res = requests.get(url).json()
            if "properties" not in res: return None
            data = res["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
            irradiance_hourly = []
            keys = sorted(list(data.keys()))
            for k in keys:
                val = data[k]
                irradiance_hourly.append(0.0 if val == -999.0 else val)
            irradiance_kst = np.zeros_like(irradiance_hourly)
            irradiance_kst[9:] = irradiance_hourly[:-9]
            if np.sum(irradiance_kst) == 0: return None
            return np.array(irradiance_kst)
        except: return None

    @st.cache_data(ttl=3600)
    def fetch_smp_rec_prices(_self, target_date):
        smp_price = 120.0 
        rec_price = 60.0  
        smp_success = False
        rec_success = False
        
        try:
            trade_day = target_date.strftime("%Y%m%d")
            smp_url = f"https://apis.data.go.kr/B552115/SmpWithForecastDemand/getSmpWithForecastDemand?serviceKey={_self.public_api_key}&pageNo=1&numOfRows=24&dataType=JSON&tradeDay={trade_day}"
            res_smp = requests.get(smp_url, timeout=8)
            
            if int(res_smp.status_code) == 200:
                smp_data = res_smp.json()
                if smp_data.get('response', {}).get('header', {}).get('resultCode') == '00':
                    api_items = smp_data['response']['body']['items']['item']
                    val_list = [float(x['smp']) for x in api_items if 'smp' in x]
                    if val_list:
                        smp_price = sum(val_list) / len(val_list)
                        smp_success = True
        except Exception: pass

        try:
            end_ymd = target_date.strftime("%Y%m")
            start_ymd = (target_date - timedelta(days=365)).strftime("%Y%m")
            rec_url = f"http://www.iwest.co.kr:8082/openapi-data/service/TradeList/Trade?serviceKey={_self.public_api_key}&strDateS={start_ymd}&strDateE={end_ymd}"
            res_rec = requests.get(rec_url, timeout=8)
            
            if int(res_rec.status_code) == 200:
                parsed_xml = ET.fromstring(res_rec.content)
                price_list = []
                for node in parsed_xml.findall('.//item'):
                    p_node = node.find('recprice')
                    if p_node is not None and p_node.text:
                        price_val = float(p_node.text) / 1000.0
                        price_list.append(price_val)
                if price_list:
                    rec_price = sum(price_list) / len(price_list)
                    rec_success = True
        except Exception: pass
        
        return round(smp_price, 2), round(rec_price, 2), smp_success, rec_success

# ==========================================
# 4. 전기요금 계산기 (2026 개편안 반영)
# ==========================================
class TariffCalculator:
    def __init__(self):
        self.res_low = {
            'other': {'base': [910, 1600, 7300], 'rate': [120.0, 214.6, 307.3]},
            'summer': {'base': [910, 1600, 7300], 'rate': [120.0, 214.6, 307.3]}
        }
        self.com_general = {'base_per_kw': 6160}
        self.res_tou = {'base': 4310}

    def get_res_tou_rate(self, month, hour):
        is_summer_winter = month in [12, 1, 2, 6, 7, 8]
        if 22 <= hour or hour < 8: return 138.7 if is_summer_winter else 125.8
        elif 8 <= hour < 16: return 184.7 if is_summer_winter else 153.8
        else: return 220.5 if is_summer_winter else 172.4

    def get_com_general_rate(self, month, hour, is_weekend):
        season = 'summer' if month in [6,7,8] else ('winter' if month in [11,12,1,2] else 'spring_fall')
        is_off_peak = (23 <= hour or hour < 9)
        # 2026 개편안: 15~17시, 18~21시가 새로운 최대부하(피크)
        is_on_peak = (15 <= hour < 17) or (18 <= hour < 21)
        
        if season == 'summer': 
            rate = 167.8 if is_on_peak else (101.1 if is_off_peak else 153.8)
        elif season == 'winter': 
            rate = 158.5 if is_on_peak else (108.2 if is_off_peak else 149.4)
        else: 
            rate = 120.2 if is_on_peak else (101.1 if is_off_peak else 120.0)

        # 2026 개편안: 봄/가을 주말 11~14시 50% 반값 할인
        if season == 'spring_fall' and is_weekend and (11 <= hour < 14):
            rate *= 0.5
            
        return rate

    def calc_res_progressive(self, kwh, is_summer):
        b1, b2 = (300, 450) if is_summer else (200, 400)
        table = self.res_low['summer'] if is_summer else self.res_low['other']
        if kwh <= b1: return table['base'][0] + kwh * table['rate'][0]
        elif kwh <= b2: return table['base'][1] + (b1 * table['rate'][0]) + ((kwh-b1) * table['rate'][1])
        else:
            cost = table['base'][2] + (b1 * table['rate'][0]) + ((b2-b1) * table['rate'][1]) + ((kwh-b2) * table['rate'][2])
            if kwh > 1000: cost += (kwh - 1000) * (736.2 - table['rate'][2])
            return cost

    def calculate_cost(self, tariff_type, timestamps, net_powers_kwh, base_monthly_kwh=0, contract_kw=3):
        total_hp_kwh = sum(net_powers_kwh)
        days = max((timestamps[-1] - timestamps[0]).days + 1, 1)
        scale_factor = 30 / days if days < 30 else 1.0
        
        estimated_monthly_hp_kwh = total_hp_kwh * scale_factor
        is_summer = timestamps[0].month in [7, 8]
        
        if tariff_type == "res_progressive":
            base_cost = self.calc_res_progressive(base_monthly_kwh, is_summer)
            final_cost = self.calc_res_progressive(base_monthly_kwh + estimated_monthly_hp_kwh, is_summer)
            return {"type": "주택용 전력(누진제)", "total_cost": int(final_cost), "hp_pure_cost": int(final_cost - base_cost)}
        elif tariff_type == "res_tou":
            usage_charge = sum(p * self.get_res_tou_rate(t.month, t.hour) for t, p in zip(timestamps, net_powers_kwh))
            monthly_usage = usage_charge * scale_factor
            return {"type": "주택용 계시별 요금제", "total_cost": int(self.res_tou['base'] + monthly_usage), "hp_pure_cost": int(monthly_usage)}
        elif tariff_type == "com_general":
            base_cost = self.com_general['base_per_kw'] * contract_kw
            usage_charge = sum(p * self.get_com_general_rate(t.month, t.hour, t.weekday() >= 5) for t, p in zip(timestamps, net_powers_kwh))
            monthly_usage = usage_charge * scale_factor
            return {"type": "일반용 전력(갑) 2026 개편안", "total_cost": int(base_cost + monthly_usage), "hp_pure_cost": int(monthly_usage)}

# ==========================================
# 5. 물리 시뮬레이션 모델 런타임
# ==========================================
def get_mains_temperature(day_of_year, region_factor=1.0):
    avg_temp, amplitude = 13.0, 10.0
    if region_factor > 1.05: avg_temp, amplitude = 10.0, 11.0
    elif region_factor < 0.8: avg_temp, amplitude = 16.0, 7.0
    return avg_temp + amplitude * np.sin((2 * np.pi / 365) * (day_of_year - 35 - (365/4)))

def run_simulation(weather_temps, pv_irradiance_hourly, dhw_schedules_full, heating_hours, cfg, start_date, hp_model_obj):
    dt = 60 # 1분
    steps = len(weather_temps)
    
    ua, heat_cap, vol_tank = cfg['ua_house'], cfg['heat_cap_house'], cfg['vol_tank']
    base_target_temp, hysteresis = cfg['target_temp'], cfg['hysteresis']
    target_room_temp = cfg.get('target_room_temp', 23.0)
    region_factor, op_mode = cfg.get('region_factor', 1.0), cfg.get('op_mode', 'simultaneous')
    heating_coef = cfg.get('heating_coef', 600.0)
    
    opt_schedule = cfg.get('opt_schedule', False)
    use_weekend_schedule = cfg.get('use_weekend_schedule', False)
    kr_holidays = holidays.KR(years=range(2020, 2031))
    nominal_kw = hp_model_obj.nominal_kw
    
    pv_installed, pv_capacity = cfg.get('pv_installed', False), cfg.get('pv_capacity', 3)
    pv_dir_multiplier = 1.0 if cfg.get('pv_direction', '남향') == '남향' else 0.85
    use_fallback_pv = (pv_irradiance_hourly is None)
    
    cp_water, temp_tank, temp_room = 4186, 50.0, 20.0
    
    res = {'time': [], 'tank': [], 'room': [], 'amb': [], 'cop': [], 'inlet': [], 'dhw_active': [], 'heat_active': [], 'power_kwh': [], 'timestamps': [], 'net_power_kwh': [], 'pv_self_kwh': [], 'pv_export_kwh': [], 'status': []}
    hp_running = False
    
    dhw_specs = {'shower': {'flow': 10, 'temp': 40}, 'wash': {'flow': 6, 'temp': 38}, 'sink': {'flow': 5, 'temp': 45}, 'laundry': {'flow': 12, 'temp': 40}}
    base_dt = datetime.combine(start_date, datetime.min.time())
    
    for t in range(steps):
        current_dt = base_dt + timedelta(minutes=t)
        t_amb = weather_temps[t]
        hour, minute = current_dt.hour, current_dt.minute
        month = current_dt.month
        
        is_weekend = (current_dt.weekday() >= 5) or (current_dt.date() in kr_holidays)
        t_inlet = get_mains_temperature(current_dt.timetuple().tm_yday, region_factor)
        
        # --- PV 발전량 계산 ---
        pv_gen_kw_min = 0.0
        if pv_installed:
            if not use_fallback_pv and (t // 60) < len(pv_irradiance_hourly):
                pv_gen_kw = (pv_irradiance_hourly[t // 60] / 1000.0) * pv_capacity * 0.8 * pv_dir_multiplier
            else:
                if 8 <= hour <= 18:
                    hour_float = hour + (minute / 60.0)
                    irradiance = 800.0 * (1.0 - ((hour_float - 13.0)**2 / 25.0))
                    pv_gen_kw = (max(0.0, irradiance) / 1000.0) * pv_capacity * 0.8 * pv_dir_multiplier
                else:
                    pv_gen_kw = 0.0
            pv_gen_kw_min = pv_gen_kw * (1/60)
        
        # --- 부하 계산 ---
        q_dhw, dhw_on = 0, False
        current_schedule_dict = dhw_schedules_full['weekend'] if is_weekend and use_weekend_schedule else dhw_schedules_full['weekday']
        
        for item, schedules in current_schedule_dict.items():
            for s in schedules:
                if s['enabled'] and (s['hour'] * 60 + s['minute'] <= hour * 60 + minute < s['hour'] * 60 + s['minute'] + s['duration']):
                    dhw_on = True
                    target_use = dhw_specs[item]['temp']
                    actual_flow = dhw_specs[item]['flow'] * ((target_use - t_inlet) / max(temp_tank - t_inlet, 1.0)) if temp_tank > target_use else dhw_specs[item]['flow']
                    q_dhw += (actual_flow / 60) * cp_water * (temp_tank - t_inlet) * dt

        q_heating, heat_on = 0, False
        q_loss = ua * (temp_room - t_amb) * dt
        if op_mode == 'priority' and dhw_on: pass
        elif hour in heating_hours and temp_room < target_room_temp:
            heat_on, q_heating = True, heating_coef * (temp_tank - temp_room) * dt * 0.95
            
        q_tank_loss = 1.84 * (temp_tank - 20.0) * dt
        
        dynamic_target = base_target_temp
        if opt_schedule:
            if month in [3, 4, 5, 9, 10] and is_weekend and (11 <= hour < 14):
                dynamic_target = min(75.0, base_target_temp + 10)
            elif 18 <= hour < 21:
                dynamic_target = max(40.0, base_target_temp - 5)

        if temp_tank < dynamic_target - hysteresis: hp_running = True
        elif temp_tank >= dynamic_target: hp_running = False
            
        q_hp, input_power_kw, op_status = 0, 0, 'Off'
        if hp_running:
            cap_w, cop_val = hp_model_obj.get_performance(t_amb, temp_tank)
            
            if cap_w <= 0:
                if cfg.get('use_backup_heater', True):
                    q_hp = (nominal_kw * 1000) * dt
                    input_power_kw = nominal_kw
                    op_status = 'Backup Heater'
                else:
                    q_hp = 0.0
                    input_power_kw = 0.0
                    op_status = 'Limit Stopped'
            else:
                q_hp = cap_w * dt
                input_power_kw = (cap_w / cop_val) / 1000
                op_status = 'Heat Pump'
                
        temp_tank += (q_hp - q_heating - q_dhw - q_tank_loss) / (vol_tank * cp_water)
        temp_room += (q_heating - q_loss) / heat_cap
        
        hp_kwh_step = input_power_kw * (1/60)
        
        self_consumed = min(hp_kwh_step, pv_gen_kw_min)
        net_hp_kwh = hp_kwh_step - self_consumed
        exported_kwh = pv_gen_kw_min - self_consumed
        
        res['time'].append(t/60)
        res['tank'].append(temp_tank)
        res['room'].append(temp_room)
        res['amb'].append(t_amb)
        res['inlet'].append(t_inlet)
        res['dhw_active'].append(dhw_on)
        res['heat_active'].append(heat_on)
        res['power_kwh'].append(hp_kwh_step)
        res['timestamps'].append(current_dt)
        res['net_power_kwh'].append(net_hp_kwh)
        res['pv_self_kwh'].append(self_consumed)
        res['pv_export_kwh'].append(exported_kwh)
        res['status'].append(op_status)
        
    res['pv_fallback_used'] = use_fallback_pv
    return res

def analyze_cold_events(res, dhw_schedules_full, start_date, cfg):
    warnings = {k: [] for k in ['shower', 'wash', 'sink', 'laundry']}
    df = pd.DataFrame({'time': res['timestamps'], 'tank': res['tank']})
    act_map = {'shower': '샤워', 'wash': '세면', 'sink': '설거지', 'laundry': '세탁기'}
    total_days = (res['timestamps'][-1].date() - res['timestamps'][0].date()).days + 1
    kr_holidays = holidays.KR(years=range(2020, 2031))
    
    for d in range(total_days):
        current_date = start_date + timedelta(days=d)
        is_weekend = (current_date.weekday() >= 5) or (current_date in kr_holidays)
        current_schedule_dict = dhw_schedules_full['weekend'] if is_weekend and cfg.get('use_weekend_schedule', False) else dhw_schedules_full['weekday']
        
        for activity_key, settings in current_schedule_dict.items():
            for s in settings:
                if not s['enabled']: continue
                start_dt = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=s['hour'], minutes=s['minute'])
                end_dt = start_dt + timedelta(minutes=s['duration'])
                
                mask = (df['time'] >= start_dt) & (df['time'] < end_dt)
                subset = df.loc[mask]
                
                if subset.empty: continue
                min_temp = subset['tank'].min()
                if min_temp < 39.0:
                    min_row = subset.loc[subset['tank'].idxmin()]
                    time_str = min_row['time'].strftime("%Y-%m-%d %H:%M")
                    warnings[activity_key].append({'time': time_str, 'temp': min_temp, 'msg': f"{time_str} '{act_map[activity_key]}' 중 발생! 온도는 '{min_temp:.1f}도'"})
    return warnings

# ==========================================
# 6. Streamlit UI
# ==========================================
st.set_page_config(page_title="스마트 듀얼 히트펌프 시뮬레이터", layout="wide")
st.title("🏡 스마트 듀얼 냉매 비교 시뮬레이터 (R290 vs R32)")

# 1. 듀얼 히트펌프 매치업 (Match-up) 설정
st.subheader("1. 듀얼 히트펌프 매치업 (Match-up) 설정")
col_hp1, col_vs, col_hp2 = st.columns([2, 0.5, 2])
with col_hp1:
    st.info("🔵 기준 모델 (R290)")
    ref1 = "R290"
    cap1 = st.selectbox("용량 1 (kW)", [12, 14, 22], index=0)
with col_vs:
    st.markdown("<h2 style='text-align: center; margin-top: 30px;'>VS</h2>", unsafe_allow_html=True)
with col_hp2:
    st.success("🟢 비교 모델 (R32)")
    ref2 = "R32"
    cap2 = st.selectbox("용량 2 (kW)", [9, 12, 16], index=1)

st.divider()

# (1) 환경 설정
col1, col2, col3, col4 = st.columns([1.5, 1.5, 1, 1])
with col1: selected_sido = st.selectbox("시/도 선택", list(KOREA_LOCATIONS.keys()))
with col2: selected_gungu = st.selectbox("시/군/구 선택", KOREA_LOCATIONS.get(selected_sido, []))
full_address = f"{selected_sido} {selected_gungu}"
auto_region_factor = get_climate_region(selected_sido, selected_gungu)
kdhc_region = get_kdhc_region(selected_sido, selected_gungu)

with col3: start_date = st.date_input("📅 시작일", datetime(2023, 1, 15))
with col4: end_date = st.date_input("📅 종료일", datetime(2023, 1, 17))

st.divider()

# (2) 주택 및 단열 정보
st.subheader("2. 주택 및 설비 정보")
col_h1, col_h2, col_h3, col_h4, col_h5 = st.columns(5)
with col_h1: year_opt = st.selectbox("건축년도", options=[1.4, 0.7, 0.35], format_func=lambda x: "1980~2000년" if x==1.4 else ("2001~2015년" if x==0.7 else "2016년 이후"))
with col_h2: type_opt = st.selectbox("주택형태", options=['apt', 'house'], format_func=lambda x: "아파트" if x=='apt' else "단독주택")
with col_h3: area_val = st.number_input("전용면적(㎡)", value=85, step=1)

if type_opt == 'apt':
    with col_h4:
        reg_opts = [1.1, 1.0, 0.85, 0.7]
        idx = reg_opts.index(auto_region_factor) if auto_region_factor in reg_opts else 1
        region_opt = st.selectbox("지역계수", options=reg_opts, index=idx, format_func=lambda x: "중부1" if x==1.1 else ("중부2" if x==1.0 else ("남부" if x==0.85 else "제주")))
    with col_h5: 
        pos_opt = st.selectbox("위치", options=['mid', 'top', 'bot'], format_func=lambda x: "중간층" if x=='mid' else ("최상층" if x=='top' else "최하층"))
else:
    region_opt = auto_region_factor
    pos_opt = 'mid' 

exposed = (np.sqrt(area_val)*4*2.5*0.4) if type_opt == 'apt' else (np.sqrt(area_val)*4*2.5 + area_val*2) 
if type_opt == 'apt' and pos_opt == 'top': exposed += area_val
elif type_opt == 'apt' and pos_opt == 'bot': exposed += area_val*0.7
final_ua = year_opt * region_opt * exposed
heat_cap = area_val * 200000

unit_load_w_m2 = get_unit_heat_load(kdhc_region, year_opt, area_val, type_opt)
total_heat_w = unit_load_w_m2 * area_val
calculated_heating_coef = (total_heat_w / 25.0) * 1.25 
st.info(f"💡 **난방 설계 부하 자동 산출:** 현재 선택된 조건(지역: {kdhc_region})에 따라 바닥 방열계수는 **{calculated_heating_coef:.1f} W/K**로 정밀 적용됩니다.")

st.divider()

# (3) 요금제 및 태양광 설정
st.subheader("3. 전기요금 및 태양광 설정 (2026 개편안 반영)")
col_t1, col_t2, col_t3 = st.columns(3)

with col_t1:
    tariff_option = st.selectbox("요금제 선택 (최신 기준)", 
                                 ["일반용 전력(갑) I (추천/개편안 적용)", "주택용 전력(누진제)", "주택용 계절·시간대별 선택요금"])
    if tariff_option == "주택용 전력(누진제)": tariff_code = "res_progressive"
    elif tariff_option == "주택용 계절·시간대별 선택요금": tariff_code = "res_tou"
    else: tariff_code = "com_general"

with col_t2:
    if tariff_code == "res_progressive":
        base_monthly_kwh = st.selectbox("기존 월간 전기 사용량 (kWh)", [0, 100, 200, 300, 400, 500, 600, 800, 1000], index=2)
        contract_kw = 3
    else:
        contract_kw = st.number_input("계약 전력 (kW)", value=3, disabled=(tariff_code == "res_tou"))
        base_monthly_kwh = 0

with col_t3:
    if tariff_code == "com_general": st.success("💡 **최신 정책 반영:** 전국 육지에서도 가장 유리한 '일반용 요금 분리 적용'이 가능합니다.")
    elif tariff_code == "res_tou": st.success("💡 **최신 정책 반영:** 과거 제주도만 적용되던 계시별 요금제를 육지에서도 선택 가능합니다.")
    else: st.warning("⚡ **누진제 유의:** 태양광 발전량이 충분하지 않다면 누진 구간 폭탄에 주의하세요.")

st.markdown("---")
use_pv = st.checkbox("☀️ 태양광(PV) 패널 설치 유무 (API 단가 자동 조회)", value=False)
if use_pv:
    c_pv1, c_pv2, c_pv3 = st.columns(3)
    with c_pv1: pv_cap = st.selectbox("패널 용량 (kW)", [3, 4, 5, 7, 10], index=0)
    with c_pv2: pv_dir = st.selectbox("패널 방향", ["남향", "동향", "서향"], index=0)
    with c_pv3: pv_date = st.date_input("설치 시작일 (매전 단가 조회용)", datetime(2023, 1, 15))

st.divider()

# (4) 온수 및 설비 (스마트 로직 & 주말 패턴 분리)
st.subheader("4. 히트펌프 운영 및 온수 스케줄 설정")
col_op, col_smart = st.columns([2, 2])
with col_op:
    op_mode_select = st.radio("운전 모드 선택", ["난방/온수 동시 열 공급", "난방/온수전환 열 공급(3-Way 밸브사용)"])
    final_op_mode = 'priority' if "3-Way" in op_mode_select else 'simultaneous'
with col_smart:
    st.markdown("**지능형 스케줄 제어 & 극한 조건 연동**")
    opt_schedule_enabled = st.checkbox("🧠 요금제 맞춤형 제어 로직 (스마트 축열 및 피크 회피)", value=True)
    use_backup_heater = st.checkbox("🔥 극한 조건/고온수 백업 전기히터 가동 허용", value=True)
    st.caption("✅ 할인 시간대 온수 축열 / ✅ 최대부하 가동 회피\n✅ 한계 도달 시 히터 개입 여부")

def synced_input(label, min_v, max_v, def_v):
    k = f"sl_{label.replace(' ', '_')}"
    if k not in st.session_state: st.session_state[k] = def_v
    st.slider(label, min_v, max_v, key=k)
    return st.session_state[k]

c_eq1, c_eq2 = st.columns(2)
with c_eq1:
    vol_tank = synced_input("물탱크 용량(L)", 100, 600, 300)
    target_temp = synced_input("물탱크 기본 목표온도(℃)", 40, 75, 50)
    hysteresis = synced_input("재가동 온도차(℃)", 2, 20, 5)
with c_eq2:
    target_room_temp = synced_input("실내 목표온도(℃)", 18, 30, 23)
    heating_hours = st.multiselect("난방 가동 시간", list(range(24)), default=[22,23,0,1,2,3,4,5,6], format_func=lambda x: f"{x}시")

if 'dhw_counts_wk' not in st.session_state: st.session_state.dhw_counts_wk = {'shower': 1, 'wash': 1, 'sink': 1, 'laundry': 1}
if 'dhw_counts_we' not in st.session_state: st.session_state.dhw_counts_we = {'shower': 1, 'wash': 1, 'sink': 1, 'laundry': 1}

dhw_schedules_full = {'weekday': {}, 'weekend': {}}

def multi_dhw_ui(label, key_name, def_h, def_m, def_d, dict_key, counts_state_key):
    schedules = []
    with st.expander(f"{label} 설정 ({st.session_state[counts_state_key][key_name]}건)"):
        for i in range(st.session_state[counts_state_key][key_name]):
            c1, c2, c3, c4 = st.columns([0.8, 1.5, 1.5, 1.5])
            with c1: en = st.checkbox("On", True, key=f"{dict_key}_{key_name}_{i}_en")
            with c2: h = st.number_input("시", 0, 23, def_h, key=f"{dict_key}_{key_name}_{i}_h")
            with c3: m = st.selectbox("분", range(0, 60, 5), index=def_m//5, key=f"{dict_key}_{key_name}_{i}_m")
            with c4: d = st.selectbox("분(사용)", range(5, 65, 5), index=(def_d//5)-1, key=f"{dict_key}_{key_name}_{i}_d")
            schedules.append({'enabled': en, 'hour': h, 'minute': m, 'duration': d})
        if st.button("추가 +", key=f"btn_{dict_key}_{key_name}"): 
            st.session_state[counts_state_key][key_name] += 1
            st.rerun()
    return schedules

use_weekend_schedule = st.checkbox("☑️ 주말 및 공휴일 스케줄 별도 설정", value=True)

if use_weekend_schedule:
    tab1, tab2 = st.tabs(["📅 평일 패턴", "🎉 주말 및 공휴일 패턴"])
    
    with tab1:
        st.caption("월~금 평일의 온수 사용 패턴입니다.")
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        with col_d1: dhw_schedules_full['weekday']['shower'] = multi_dhw_ui("🚿 샤워", "shower", 7, 0, 20, "wk", "dhw_counts_wk")
        with col_d2: dhw_schedules_full['weekday']['wash'] = multi_dhw_ui("🪥 세면", "wash", 8, 10, 10, "wk", "dhw_counts_wk")
        with col_d3: dhw_schedules_full['weekday']['sink'] = multi_dhw_ui("🍽️ 설거지", "sink", 19, 30, 20, "wk", "dhw_counts_wk")
        with col_d4: dhw_schedules_full['weekday']['laundry'] = multi_dhw_ui("👕 세탁기", "laundry", 10, 0, 40, "wk", "dhw_counts_wk")
        
    with tab2:
        st.caption("주말(토/일) 및 법정 공휴일에 작동하는 패턴입니다.")
        col_w1, col_w2, col_w3, col_w4 = st.columns(4)
        with col_w1: dhw_schedules_full['weekend']['shower'] = multi_dhw_ui("🚿 샤워", "shower", 9, 0, 20, "we", "dhw_counts_we")
        with col_w2: dhw_schedules_full['weekend']['wash'] = multi_dhw_ui("🪥 세면", "wash", 9, 30, 10, "we", "dhw_counts_we")
        with col_w3: dhw_schedules_full['weekend']['sink'] = multi_dhw_ui("🍽️ 설거지", "sink", 14, 0, 20, "we", "dhw_counts_we")
        with col_w4: dhw_schedules_full['weekend']['laundry'] = multi_dhw_ui("👕 세탁기", "laundry", 11, 0, 40, "we", "dhw_counts_we")
else:
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    with col_d1: dhw_schedules_full['weekday']['shower'] = multi_dhw_ui("🚿 샤워", "shower", 7, 0, 20, "wk", "dhw_counts_wk")
    with col_d2: dhw_schedules_full['weekday']['wash'] = multi_dhw_ui("🪥 세면", "wash", 8, 10, 10, "wk", "dhw_counts_wk")
    with col_d3: dhw_schedules_full['weekday']['sink'] = multi_dhw_ui("🍽️ 설거지", "sink", 19, 30, 20, "wk", "dhw_counts_wk")
    with col_d4: dhw_schedules_full['weekday']['laundry'] = multi_dhw_ui("👕 세탁기", "laundry", 10, 0, 40, "wk", "dhw_counts_wk")
    dhw_schedules_full['weekend'] = dhw_schedules_full['weekday']

# (5) 실행
st.markdown("---")
if st.button("🚀 듀얼 시뮬레이션 및 경제성 비교 시작", type="primary", use_container_width=True):
    dm = DataManager()
    
    if use_pv:
        with st.spinner("한국전력거래소 SMP/REC API 조회 중..."):
            smp_val, rec_val, smp_ok, rec_ok = dm.fetch_smp_rec_prices(pv_date)
            st.session_state.api_smp = smp_val
            st.session_state.api_rec = rec_val
            st.session_state.smp_ok = smp_ok
            st.session_state.rec_ok = rec_ok

    with st.spinner("기상 데이터 연동 및 듀얼 엔진 가동 중..."):
        lat, lon = dm.get_coordinates(full_address)
        if lat is None: 
            st.error("주소를 찾을 수 없습니다.")
            st.stop()
            
        temps, err = dm.get_weather_data(lat, lon, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        if err: 
            st.error(err)
            st.stop()
            
        pv_irradiance = dm.get_solar_irradiance(lat, lon, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")) if use_pv else None

    cfg = {
        'ua_house': final_ua, 'heat_cap_house': heat_cap, 'vol_tank': vol_tank, 
        'target_temp': target_temp, 'hysteresis': hysteresis, 'target_room_temp': target_room_temp, 
        'region_factor': region_opt, 'op_mode': final_op_mode,
        'heating_coef': calculated_heating_coef,
        'opt_schedule': opt_schedule_enabled,
        'use_weekend_schedule': use_weekend_schedule,
        'use_backup_heater': use_backup_heater,
        'pv_installed': use_pv, 'pv_capacity': pv_cap if use_pv else 0, 'pv_direction': pv_dir if use_pv else '남향'
    }
    
    # 듀얼 엔진 객체 생성 및 동시 실행
    hp1_model = RealHeatPumpModel(ref1, cap1)
    hp2_model = RealHeatPumpModel(ref2, cap2)
    
    res1 = run_simulation(temps, pv_irradiance, dhw_schedules_full, heating_hours, cfg, start_date, hp1_model)
    res2 = run_simulation(temps, pv_irradiance, dhw_schedules_full, heating_hours, cfg, start_date, hp2_model)
    
    tc = TariffCalculator()
    
    cost_res1 = tc.calculate_cost(tariff_code, res1['timestamps'], res1['net_power_kwh'], base_monthly_kwh, contract_kw)
    cost_no_pv1 = tc.calculate_cost(tariff_code, res1['timestamps'], res1['power_kwh'], base_monthly_kwh, contract_kw)
    pv_savings_krw1 = cost_no_pv1['hp_pure_cost'] - cost_res1['hp_pure_cost']
    
    cost_res2 = tc.calculate_cost(tariff_code, res2['timestamps'], res2['net_power_kwh'], base_monthly_kwh, contract_kw)
    cost_no_pv2 = tc.calculate_cost(tariff_code, res2['timestamps'], res2['power_kwh'], base_monthly_kwh, contract_kw)
    pv_savings_krw2 = cost_no_pv2['hp_pure_cost'] - cost_res2['hp_pure_cost']
    
    st.session_state.res1 = res1
    st.session_state.res2 = res2
    st.session_state.cost1 = cost_res1
    st.session_state.cost2 = cost_res2
    st.session_state.pv_savings_krw1 = pv_savings_krw1
    st.session_state.pv_savings_krw2 = pv_savings_krw2
    st.session_state.last_cfg = cfg
    st.session_state.last_start = start_date
    st.session_state.ref1 = ref1
    st.session_state.cap1 = cap1
    st.session_state.ref2 = ref2
    st.session_state.cap2 = cap2
    st.rerun()

# (6) 듀얼 시뮬레이션 결과 화면
if 'res1' in st.session_state:
    res1 = st.session_state.res1
    res2 = st.session_state.res2
    cost1 = st.session_state.cost1
    cost2 = st.session_state.cost2
    cfg = st.session_state.last_cfg
    
    st.header("🏆 듀얼 히트펌프 경제성 및 효율 비교 (월간 환산)")
    
    api_warnings = []
    if cfg['pv_installed']:
        if res1.get('pv_fallback_used'):
            api_warnings.append("NASA 일조량 API: 데이터 지연으로 인해 '표준 맑은 날 가상 커브' 자동 적용")
        if not st.session_state.get('smp_ok', False):
            api_warnings.append("한국전력 SMP API: 응답 지연으로 기본 단가(120원) 적용")
        if not st.session_state.get('rec_ok', False):
            api_warnings.append("서부발전 REC API: 응답 지연으로 기본 단가(60원) 적용")
            
    if api_warnings:
        for w in api_warnings: st.warning(f"📡 {w}")
    
    days_simulated = max((res1['timestamps'][-1] - res1['timestamps'][0]).days + 1, 1)
    scale = 30 / days_simulated if days_simulated < 30 else 1.0
    
    c_res1, c_res2 = st.columns(2)
    with c_res1:
        st.info(f"🔵 **기준 모델: {st.session_state.ref1} ({st.session_state.cap1}kW)**")
        if cfg['pv_installed']:
            total_export_month = sum(res1['pv_export_kwh']) * scale
            pv_revenue_month = int(total_export_month * (st.session_state.api_smp + st.session_state.api_rec)) 
            real_net_cost = cost1['hp_pure_cost'] - pv_revenue_month
            st.metric("실제 전기 체감비용", f"{real_net_cost:,} 원", "청구액 - 매전수익", delta_color="inverse")
            st.metric("청구될 히트펌프 요금", f"{cost1['hp_pure_cost']:,} 원", f"PV 자가소비 절감액: -{st.session_state.pv_savings_krw1:,}원")
        else:
            st.metric("월간 예상 전기료", f"{cost1['total_cost']:,} 원", delta=f"+{cost1['hp_pure_cost']:,}원 (HP 단독분)")
            st.metric("히트펌프 월 소비전력", f"{sum(res1['power_kwh']) * scale:.1f} kWh")
            
    with c_res2:
        st.success(f"🟢 **비교 모델: {st.session_state.ref2} ({st.session_state.cap2}kW)**")
        if cfg['pv_installed']:
            total_export_month2 = sum(res2['pv_export_kwh']) * scale
            pv_revenue_month2 = int(total_export_month2 * (st.session_state.api_smp + st.session_state.api_rec)) 
            real_net_cost2 = cost2['hp_pure_cost'] - pv_revenue_month2
            st.metric("실제 전기 체감비용", f"{real_net_cost2:,} 원", delta=f"{real_net_cost2 - real_net_cost:,} 원 (기준 대비)", delta_color="inverse")
            st.metric("청구될 히트펌프 요금", f"{cost2['hp_pure_cost']:,} 원")
        else:
            diff_cost = cost2['total_cost'] - cost1['total_cost']
            diff_power = (sum(res2['power_kwh']) - sum(res1['power_kwh'])) * scale
            st.metric("월간 예상 전기료", f"{cost2['total_cost']:,} 원", delta=f"{diff_cost:,} 원 (기준 대비)", delta_color="inverse")
            st.metric("히트펌프 월 소비전력", f"{sum(res2['power_kwh']) * scale:.1f} kWh", delta=f"{diff_power:.1f} kWh")

    st.caption("※ 위 요금은 시뮬레이션 기간의 패턴이 한 달간 지속된다고 가정하여 30일치로 환산한 추정치입니다.")
    st.divider()

    st.subheader("📊 운전 패턴 및 온도 변화 (스마트 제어 모니터링)")
    
    def render_warnings(res_data, model_name):
        warnings_dict = analyze_cold_events(res_data, dhw_schedules_full, st.session_state.last_start, cfg)
        priority_order = ['shower', 'wash', 'sink', 'laundry']
        act_labels = {'shower': '🚿 샤워', 'wash': '🪥 세면', 'sink': '🍽️ 설거지', 'laundry': '👕 세탁기'}
        has_warning = False

        for key in priority_order:
            items = warnings_dict.get(key, [])
            if not items: continue
            has_warning = True
            label = act_labels[key]

            if len(items) > 1:
                sorted_items = sorted(items, key=lambda x: x['temp'])
                worst = sorted_items[0]
                st.error(f"⚠️ [{model_name} 대표 발생] {worst['msg']} (외 {len(items)-1}건)")
                with st.expander(f"🔻 {model_name} {label} 온수 부족 전체 내역 보기"):
                    for item in sorted_items: st.write(f"- {item['msg']}")
            else:
                st.error(f"⚠️ [{model_name}] {items[0]['msg']}")

        if not has_warning:
            st.success(f"✅ [{model_name}] 온수 사용 중 물탱크 온도가 39도 미만으로 떨어진 적이 없습니다.")

    w_col1, w_col2 = st.columns(2)
    with w_col1:
        st.markdown(f"**🔵 {st.session_state.ref1} ({st.session_state.cap1}kW) 온수 알림**")
        render_warnings(res1, st.session_state.ref1)
    with w_col2:
        st.markdown(f"**🟢 {st.session_state.ref2} ({st.session_state.cap2}kW) 온수 알림**")
        render_warnings(res2, st.session_state.ref2)

    st.markdown("<br>", unsafe_allow_html=True)
    
    tab_r290, tab_r32, tab_comp = st.tabs([
        f"🔵 기준 모델 상세 ({st.session_state.ref1})", 
        f"🟢 비교 모델 상세 ({st.session_state.ref2})", 
        "🏆 통합 비교 분석"
    ])
    
    plot_times = res1['timestamps']
    
    def plot_single_res(ax1, res_data, ref_name, plot_opts, color_theme='blue'):
        ax1.plot(plot_times, res_data['tank'], color=color_theme, linestyle='-', label=f'Tank Temp ({ref_name})', lw=1.5)
        ax1.plot(plot_times, res_data['room'], 'r-', label='Room Temp', lw=2)
        ax1.plot(plot_times, res_data['amb'], 'g--', label='Outdoor', alpha=0.4)
        
        if "찬물(급수) 온도" in plot_opts: 
            ax1.plot(plot_times, res_data['inlet'], color='purple', linestyle=':', label='Inlet Water', alpha=0.6)
            
        ymin, ymax = ax1.get_ylim()
        if "난방 가동구간" in plot_opts:
            heat_arr = np.array(res_data['heat_active'])
            ax1.fill_between(plot_times, ymin, ymax, where=heat_arr, color='red', alpha=0.1, label='Heating Active')
        if "온수 가동구간" in plot_opts:
            dhw_arr = np.array(res_data['dhw_active'])
            ax1.fill_between(plot_times, ymin, ymax, where=dhw_arr, color='blue', alpha=0.1, label='DHW Active')

        ax1.set_ylabel("Temperature (C)")
        ax1.legend(loc='upper left', ncol=3)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%Hh'))
        
        if "히트펌프 소비전력(kW)" in plot_opts or "태양광 발전량(kW)" in plot_opts:
            ax2 = ax1.twinx()
            if "히트펌프 소비전력(kW)" in plot_opts:
                ax2.fill_between(plot_times, 0, np.array(res_data['power_kwh']) * 60, color='orange', alpha=0.3, label=f'HP Power (kW)')
                backup = np.array([1 if 'Backup' in s else 0 for s in res_data['status']])
                if np.any(backup):
                    ax2.fill_between(plot_times, 0, np.array(res_data['power_kwh']) * 60, where=(backup==1), color='black', alpha=0.4, label='Backup Heater')

            if "태양광 발전량(kW)" in plot_opts and cfg['pv_installed']:
                pv_gen = (np.array(res_data['pv_self_kwh']) + np.array(res_data['pv_export_kwh'])) * 60
                ax2.plot(plot_times, pv_gen, color='green', lw=1.5, alpha=0.7, label='PV Generation (kW)')
                
            ax2.set_ylabel("Power Input (kW)")
            ax2.legend(loc='upper right')

    base_opts = ["난방 가동구간", "온수 가동구간", "찬물(급수) 온도", "히트펌프 소비전력(kW)"]
    base_def_opts = ["난방 가동구간", "온수 가동구간", "히트펌프 소비전력(kW)"]
    if cfg['pv_installed']:
        base_opts.append("태양광 발전량(kW)")
        base_def_opts.append("태양광 발전량(kW)")

    with tab_r290:
        opts_r290 = st.multiselect(f"[{st.session_state.ref1}] 표시할 항목 선택", options=base_opts, default=base_def_opts, key="opt_r290")
        fig1, ax1_1 = plt.subplots(figsize=(12, 6))
        plot_single_res(ax1_1, res1, st.session_state.ref1, opts_r290, color_theme='blue')
        st.pyplot(fig1)

    with tab_r32:
        opts_r32 = st.multiselect(f"[{st.session_state.ref2}] 표시할 항목 선택", options=base_opts, default=base_def_opts, key="opt_r32")
        fig2, ax1_2 = plt.subplots(figsize=(12, 6))
        plot_single_res(ax1_2, res2, st.session_state.ref2, opts_r32, color_theme='cyan')
        st.pyplot(fig2)

    with tab_comp:
        comp_opts = ["히트펌프 소비전력(kW) 비교", "난방/온수 가동구간"]
        comp_def_opts = ["히트펌프 소비전력(kW) 비교"]
        opts_comp = st.multiselect("통합 그래프 표시 항목", options=comp_opts, default=comp_def_opts, key="opt_comp")
        
        fig_comp, ax_comp1 = plt.subplots(figsize=(12, 6))
        
        ax_comp1.plot(plot_times, res1['room'], 'r-', label='Room Temp', lw=2)
        ax_comp1.plot(plot_times, res1['amb'], 'g--', label='Outdoor Temp', alpha=0.4)
        ax_comp1.plot(plot_times, res1['tank'], color='blue', linestyle='-', label=f'Tank Temp ({st.session_state.ref1})', lw=2)
        ax_comp1.plot(plot_times, res2['tank'], color='cyan', linestyle='--', label=f'Tank Temp ({st.session_state.ref2})', lw=2)
        
        ymin, ymax = ax_comp1.get_ylim()
        if "난방/온수 가동구간" in opts_comp:
            dhw_arr = np.array(res1['dhw_active'])
            heat_arr = np.array(res1['heat_active'])
            ax_comp1.fill_between(plot_times, ymin, ymax, where=heat_arr, color='red', alpha=0.05, label='Heating Active')
            ax_comp1.fill_between(plot_times, ymin, ymax, where=dhw_arr, color='blue', alpha=0.05, label='DHW Active')
            
        ax_comp1.set_ylabel("Temperature (C)")
        ax_comp1.legend(loc='upper left', ncol=2)
        ax_comp1.grid(True, alpha=0.3)
        ax_comp1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%Hh'))
        
        if "히트펌프 소비전력(kW) 비교" in opts_comp:
            ax_comp2 = ax_comp1.twinx()
            ax_comp2.plot(plot_times, np.array(res1['power_kwh']) * 60, color='red', alpha=0.6, label=f'Power {st.session_state.ref1} (kW)')
            ax_comp2.plot(plot_times, np.array(res2['power_kwh']) * 60, color='orange', alpha=0.8, linestyle=':', label=f'Power {st.session_state.ref2} (kW)')
            
            backup1 = np.array([1 if 'Backup' in s else 0 for s in res1['status']])
            backup2 = np.array([1 if 'Backup' in s else 0 for s in res2['status']])
            if np.any(backup1): ax_comp2.fill_between(plot_times, 0, np.array(res1['power_kwh']) * 60, where=(backup1==1), color='black', alpha=0.3, label=f'Backup {st.session_state.ref1}')
            if np.any(backup2): ax_comp2.fill_between(plot_times, 0, np.array(res2['power_kwh']) * 60, where=(backup2==1), color='gray', alpha=0.3, label=f'Backup {st.session_state.ref2}')
            
            ax_comp2.set_ylabel("Power Input (kW)")
            ax_comp2.legend(loc='upper right')
            
        st.pyplot(fig_comp)