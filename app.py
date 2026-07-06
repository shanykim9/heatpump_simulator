"""
=============================================================================
[스마트 히트펌프 에너지 및 요금 시뮬레이터 핵심 로직 요약]

1. 열역학 및 물리 모델
   - 히트펌프 성능 보간: 외기/출수온도에 따른 능력 및 효율(COP) 2차원 보간.
   - 제상 로직: 5℃ 미만에서 외기온도 비례 난방능력 선형 감소.
   - 실내 열평형: 주택 단열(UA), 열용량 기반의 온도 변화 예측.
   - 물탱크 열손실: 단열 성능(1.84 W/K) 적용한 자연 냉각 현상 모사.
   - 직수 온도: 계절별 지중/상수도 온도 연간 사인파 추정.

2. 제어 및 운영 알고리즘
   - 믹싱 밸브: 목표 온도와 직수 온도를 혼합비율로 계산하여 유량 산출.
   - 3-Way 밸브(급탕 우선): 온수 사용 시 난방 차단 로직.
   - 히스테리시스: 잦은 작동 방지를 위한 재가동 온도차 제어.

3. 태양광(PV) 연계 및 자동 API 알고리즘 (업데이트)
   - 발전량 예측: NASA POWER API 시간대별 일조량 활용 (실패 시 표준 맑은날 자동 생성).
   - SMP/REC 단가 자동 조회: 한국전력거래소(SMP) 및 서부발전(REC) 공공데이터 API를 
     백그라운드에서 자동 호출하여 평균 단가 추출 (REC는 1MWh 기준이므로 kWh로 환산).
     * API 응답 실패 시 시뮬레이션 중단을 막기 위해 기본값(SMP 120원/REC 60원) 자동 적용.
   - 자가소비 및 매전: 태양광 발전 전력을 우선 소비 후 잉여전력 매전 수익화 산출.

4. 최신 전기요금 산출 (2025년 4월 1일 KEPCO 기준)
   - 주택용 누진제: 하계(300/450kWh) 및 기타계절(200/400kWh) 구간별 단가 적용.
   - 일반용 전력(갑) I: 계절 및 시간대(경/중/최대부하) 차등 요금 적용.
   - 주택용 계시별 요금(제주/전국): 계절(봄가을/여름겨울) 및 3개 시간대별 차등 적용.
=============================================================================
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import RegularGridInterpolator
from geopy.geocoders import Nominatim
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import pandas as pd

# ==========================================
# 1. 행정구역 데이터
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

# ==========================================
# 2. 외부 API 관리 (Weather, NASA POWER, SMP/REC)
# ==========================================
import urllib.parse # URL 인코딩 방지를 위해 추가

class DataManager:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="hp_sim_korea_v8")
        # 제공된 통합 API Key (디코딩된 순수 문자열 형태로 변환하여 사용)
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
        
        # 1. SMP API 호출
        try:
            trade_day = target_date.strftime("%Y%m%d")
            smp_url = f"https://apis.data.go.kr/B552115/SmpWithForecastDemand/getSmpWithForecastDemand?serviceKey={_self.public_api_key}&pageNo=1&numOfRows=24&dataType=JSON&tradeDay={trade_day}"
            
            res_smp = requests.get(smp_url, timeout=8)
            
            if res_smp.status_code == 200:
                data = res_smp.json()
                if data.get('response', {}).get('header', {}).get('resultCode') == '00':
                    items = data['response']['body']['items']['item']
                    smp_values = [float(item['smp']) for item in items if 'smp' in item]
                    if smp_values:
                        smp_price = sum(smp_values) / len(smp_values)
                        smp_success = True
        except Exception:
            pass

        # 2. REC API 호출 (한국서부발전) - 핵심 수정 부분
        try:
            # [수정] REC 데이터는 월 단위 발표이므로, 최신 데이터 부재를 대비해 
            # 기준일로부터 '최근 1년간(12개월)' 범위로 넉넉하게 조회하도록 날짜 설정
            end_ymd = target_date.strftime("%Y%m") 
            start_ymd = (target_date - timedelta(days=365)).strftime("%Y%m")
            
            rec_url = f"http://www.iwest.co.kr:8082/openapi-data/service/TradeList/Trade?serviceKey={_self.public_api_key}&strDateS={start_ymd}&strDateE={end_ymd}"
            
            res_rec = requests.get(rec_url, timeout=8)
            
            if res_rec.status_code == 200:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(res_rec.content)
                prices = []
                
                # XML에서 조회된 모든 달의 'recprice'를 추출
                for item in root.findall('.//item'):
                    price_node = item.find('recprice')
                    if price_node is not None and price_node.text:
                        # 서부발전 API의 REC 가격은 1MWh 기준이므로 1000으로 나누어 kWh당 단가로 환산
                        prices.append(float(price_node.text) / 1000.0) 
                
                # 1년치 데이터가 수집되었다면 그 평균값을 REC 단가로 적용
                if prices:
                    rec_price = sum(prices) / len(prices)
                    rec_success = True
        except Exception:
            pass
        
        return round(smp_price, 2), round(rec_price, 2), smp_success, rec_success

# ==========================================
# 3. 전기요금 계산기
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

    def get_com_general_rate(self, month, hour):
        season = 'summer' if month in [6,7,8] else ('winter' if month in [11,12,1,2] else 'spring_fall')
        is_off_peak = (23 <= hour or hour < 9)
        is_on_peak = (10 <= hour < 12) or (13 <= hour < 17) 
        
        if season == 'summer': return 184.7 if is_on_peak else (96.0 if is_off_peak else 153.8)
        elif season == 'winter': return 175.4 if is_on_peak else (103.1 if is_off_peak else 149.4)
        else: return 133.4 if is_on_peak else (96.0 if is_off_peak else 120.0)

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
            usage_charge = sum(p * self.get_com_general_rate(t.month, t.hour) for t, p in zip(timestamps, net_powers_kwh))
            monthly_usage = usage_charge * scale_factor
            return {"type": "일반용 전력(갑) I", "total_cost": int(base_cost + monthly_usage), "hp_pure_cost": int(monthly_usage)}

# ==========================================
# 4. 물리 시뮬레이션 모델
# ==========================================
class RealHeatPumpModel:
    def __init__(self):
        self.amb_temps = np.array([-25, -20, -15, -10, -7, -5, 2, 7, 10, 15, 20, 25, 30, 35, 43])
        self.water_temps = np.array([35, 40, 45, 50, 55, 60, 65, 70, 75])
        
        self.capacity_data = np.array([
            [6.73, 8.15, 9.18, 10.90, 11.43, 11.46, 11.24, 14.45, 16.04, 17.41, 18.28, 17.28, 18.67, 16.65, 18.70],
            [6.60, 8.04, 9.09, 10.05, 11.07, 11.25, 11.09, 14.03, 15.72, 17.00, 17.56, 16.64, 17.98, 16.17, 18.38],
            [6.49, 7.82, 8.89, 9.89, 10.85, 11.09, 10.79, 13.76, 15.39, 16.36, 16.67, 16.15, 17.16, 15.68, 17.78],
            [5.88, 6.95, 8.22, 9.12, 9.94, 10.31, 10.45, 13.36, 15.08, 16.26, 16.38, 15.93, 16.53, 15.23, 17.48],
            [5.53, 6.66, 7.97, 9.15, 9.53, 10.19, 10.23, 12.91, 14.73, 15.89, 16.03, 14.62, 15.68, 14.46, 17.23],
            [5.00, 6.58, 7.67, 8.96, 9.45, 10.01, 10.11, 12.50, 13.73, 15.39, 15.71, 13.76, 14.94, 13.66, 16.67],
            [0.00, 0.00, 7.64, 8.83, 9.33, 9.80, 10.05, 11.92, 12.72, 11.12, 12.23, 11.83, 13.07, 12.54, 14.77],
            [0.00, 0.00, 0.00, 6.61, 7.55, 7.75, 9.06, 9.75, 10.08, 9.80, 9.97, 7.82, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 5.34, 5.57, 6.75, 7.11, 7.30, 8.05, 0.00, 0.00, 0.00, 0.00, 0.00]
        ])
        
        self.cop_data = np.array([
            [2.14, 2.57, 2.62, 3.04, 3.20, 3.15, 3.39, 3.82, 4.22, 4.51, 4.62, 6.06, 6.20, 7.43, 7.51],
            [2.02, 2.31, 2.52, 2.72, 2.92, 2.84, 2.99, 3.46, 3.82, 4.08, 4.24, 5.27, 5.47, 6.10, 6.73],
            [1.94, 2.09, 2.35, 2.48, 2.73, 2.56, 2.62, 3.28, 3.53, 3.66, 3.89, 4.40, 4.69, 5.19, 6.03],
            [1.80, 1.85, 2.23, 2.34, 2.36, 2.50, 2.27, 2.89, 3.12, 3.37, 3.44, 3.91, 4.13, 5.03, 5.41],
            [1.61, 1.81, 1.95, 2.20, 2.20, 2.27, 2.08, 2.48, 2.77, 3.09, 3.02, 3.59, 3.56, 4.22, 4.35],
            [1.37, 1.70, 1.85, 1.96, 2.06, 2.11, 2.01, 2.39, 2.50, 2.76, 2.79, 2.80, 3.02, 3.42, 3.93],
            [0.00, 0.00, 1.65, 1.83, 1.88, 1.98, 1.94, 2.20, 2.26, 2.28, 2.32, 2.43, 2.57, 3.20, 3.29],
            [0.00, 0.00, 0.00, 1.37, 1.45, 1.49, 1.67, 1.81, 1.88, 2.07, 2.09, 2.25, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 1.01, 1.03, 1.14, 1.27, 1.28, 1.38, 0.00, 0.00, 0.00, 0.00, 0.00]
        ])
        self.interp_cap = RegularGridInterpolator((self.water_temps, self.amb_temps), self.capacity_data, bounds_error=False, fill_value=None)
        self.interp_cop = RegularGridInterpolator((self.water_temps, self.amb_temps), self.cop_data, bounds_error=False, fill_value=None)

    def get_performance(self, t_amb, t_water):
        cap = self.interp_cap((min(t_water, 75.0), t_amb))
        cop = self.interp_cop((min(t_water, 75.0), t_amb))
        if cap is None or cap <= 0.1: return 0.0, 0.0
        
        if t_amb <= -25: defrost = 0.8
        elif t_amb >= 5: defrost = 1.0
        else: defrost = 0.8 + (t_amb + 25) * (0.2 / 30)
        return float(cap * 1000 * defrost), float(cop)

def get_mains_temperature(day_of_year, region_factor=1.0):
    avg_temp, amplitude = 13.0, 10.0
    if region_factor > 1.05: avg_temp, amplitude = 10.0, 11.0
    elif region_factor < 0.8: avg_temp, amplitude = 16.0, 7.0
    return avg_temp + amplitude * np.sin((2 * np.pi / 365) * (day_of_year - 35 - (365/4)))

def run_simulation(weather_temps, pv_irradiance_hourly, dhw_schedules, heating_hours, cfg, start_date):
    dt = 60 # 1분
    steps = len(weather_temps)
    
    ua, heat_cap, vol_tank = cfg['ua_house'], cfg['heat_cap_house'], cfg['vol_tank']
    target_temp, hysteresis = cfg['target_temp'], cfg['hysteresis']
    target_room_temp = cfg.get('target_room_temp', 23.0)
    region_factor, op_mode = cfg.get('region_factor', 1.0), cfg.get('op_mode', 'simultaneous')
    
    pv_installed, pv_capacity = cfg.get('pv_installed', False), cfg.get('pv_capacity', 3)
    pv_dir_multiplier = 1.0 if cfg.get('pv_direction', '남향') == '남향' else 0.85
    
    use_fallback_pv = (pv_irradiance_hourly is None)
    
    cp_water, temp_tank, temp_room = 4186, 50.0, 20.0
    
    res = {'time': [], 'tank': [], 'room': [], 'amb': [], 'cop': [], 'inlet': [], 'dhw_active': [], 'heat_active': [], 'power_kwh': [], 'timestamps': [], 'net_power_kwh': [], 'pv_self_kwh': [], 'pv_export_kwh': []}
    hp_model = RealHeatPumpModel()
    hp_running = False
    
    dhw_specs = {'shower': {'flow': 10, 'temp': 40}, 'wash': {'flow': 6, 'temp': 38}, 'sink': {'flow': 5, 'temp': 45}, 'laundry': {'flow': 12, 'temp': 40}}
    base_dt = datetime.combine(start_date, datetime.min.time())
    
    for t in range(steps):
        current_dt = base_dt + timedelta(minutes=t)
        t_amb = weather_temps[t]
        hour, minute = current_dt.hour, current_dt.minute
        t_inlet = get_mains_temperature(current_dt.timetuple().tm_yday, region_factor)
        
        # --- PV 발전량 계산 ---
        pv_gen_kw_min = 0.0
        if pv_installed:
            if not use_fallback_pv and (t // 60) < len(pv_irradiance_hourly):
                pv_gen_kw = (pv_irradiance_hourly[t // 60] / 1000.0) * pv_capacity * 0.8 * pv_dir_multiplier
            else:
                # [대체 로직] API 오류/최근 날짜일 시 표준 포물선 발전 커브 적용
                if 8 <= hour <= 18:
                    hour_float = hour + (minute / 60.0)
                    irradiance = 800.0 * (1.0 - ((hour_float - 13.0)**2 / 25.0))
                    pv_gen_kw = (max(0.0, irradiance) / 1000.0) * pv_capacity * 0.8 * pv_dir_multiplier
                else:
                    pv_gen_kw = 0.0
            
            pv_gen_kw_min = pv_gen_kw * (1/60)
        
        # --- 부하 계산 ---
        q_dhw, dhw_on = 0, False
        for item, schedules in dhw_schedules.items():
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
            heat_on, q_heating = True, 600 * (temp_tank - temp_room) * dt * 0.95
            
        q_tank_loss = 1.84 * (temp_tank - 20.0) * dt
        
        if temp_tank < target_temp - hysteresis: hp_running = True
        elif temp_tank >= target_temp: hp_running = False
            
        q_hp, input_power_kw = 0, 0
        if hp_running:
            cap_w, cop_val = hp_model.get_performance(t_amb, temp_tank)
            if cap_w <= 0: hp_running = False
            else:
                q_hp = cap_w * dt
                if cop_val > 0: input_power_kw = (cap_w / cop_val) / 1000
                
        temp_tank += (q_hp - q_heating - q_dhw - q_tank_loss) / (vol_tank * cp_water)
        temp_room += (q_heating - q_loss) / heat_cap
        
        hp_kwh_step = input_power_kw * (1/60)
        
        # 태양광 자가소비 및 잉여전력 매전 분리
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
        
    res['pv_fallback_used'] = use_fallback_pv
    return res

def analyze_cold_events(res, dhw_schedules, start_date):
    warnings = {k: [] for k in ['shower', 'wash', 'sink', 'laundry']}
    df = pd.DataFrame({'time': res['timestamps'], 'tank': res['tank']})
    act_map = {'shower': '샤워', 'wash': '세면', 'sink': '설거지', 'laundry': '세탁기'}
    total_days = (res['timestamps'][-1].date() - res['timestamps'][0].date()).days + 1
    
    for d in range(total_days):
        current_date = start_date + timedelta(days=d)
        for activity_key, settings in dhw_schedules.items():
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
# 5. Streamlit UI
# ==========================================
st.set_page_config(page_title="스마트 히트펌프 시뮬레이터", layout="wide")
st.title("🏡 스마트 히트펌프 에너지 및 경제성 시뮬레이터")

# (1) 환경 설정
col1, col2, col3, col4 = st.columns([1.5, 1.5, 1, 1])
with col1: selected_sido = st.selectbox("시/도 선택", list(KOREA_LOCATIONS.keys()))
with col2: selected_gungu = st.selectbox("시/군/구 선택", KOREA_LOCATIONS.get(selected_sido, []))
full_address = f"{selected_sido} {selected_gungu}"
auto_region_factor = get_climate_region(selected_sido, selected_gungu)

is_jeju = "제주" in selected_sido
location_type_str = "제주도" if is_jeju else "내륙"

with col3: start_date = st.date_input("📅 시작일", datetime(2023, 1, 15))
with col4: end_date = st.date_input("📅 종료일", datetime(2023, 1, 17))

st.divider()

# (2) 주택 및 단열 정보
st.subheader("1. 주택 및 설비 정보")
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

st.divider()

# (3) 요금제 및 태양광 설정
st.subheader("2. 전기요금 및 태양광 설정")
col_t1, col_t2, col_t3 = st.columns(3)

with col_t1:
    tariff_option = st.selectbox("요금제 선택 (2025년 기준)", 
                                 ["주택용 전력(누진제)", "주택용 계절·시간대별 선택요금 (계시별)", "일반용 전력(갑) I"])
    if tariff_option == "주택용 전력(누진제)": tariff_code = "res_progressive"
    elif tariff_option == "주택용 계절·시간대별 선택요금 (계시별)": tariff_code = "res_tou"
    else: tariff_code = "com_general"

with col_t2:
    if tariff_code == "res_progressive":
        base_monthly_kwh = st.selectbox("기존 월간 전기 사용량 (kWh)", [0, 100, 200, 300, 400, 500, 600, 800, 1000], index=2)
        contract_kw = 3
    else:
        contract_kw = st.number_input("일반용 계약 전력 (kW)", value=3, disabled=(tariff_code == "res_tou"))
        base_monthly_kwh = 0

with col_t3:
    if tariff_code == "res_tou": st.warning("⚡ **계시별 요금:** 봄/가을 및 여름/겨울 구간 시간대별 단가 적용")
    elif tariff_code == "res_progressive": st.success("💡 **누진제:** 히트펌프 사용량은 상위 누진구간 단가로 합산")
    else: st.info("⚡ **일반용:** 계약전력 3kW 기준 기본요금 및 계절/시간대별 단가 적용")

st.markdown("---")
# 태양광(PV) UI
use_pv = st.checkbox("☀️ 태양광(PV) 패널 설치 유무 (API 단가 자동 조회)", value=False)

if use_pv:
    c_pv1, c_pv2, c_pv3 = st.columns(3)
    with c_pv1: pv_cap = st.selectbox("패널 용량 (kW)", [3, 4, 5, 7, 10], index=0)
    with c_pv2: pv_dir = st.selectbox("패널 방향", ["남향", "동향", "서향"], index=0)
    with c_pv3: pv_date = st.date_input("설치 시작일 (매전 단가 조회용)", datetime(2023, 1, 15))

st.divider()

# (4) 온수 및 설비
st.subheader("3. 히트펌프 운영 및 온수 스케줄")
col_op, col_none = st.columns([2, 1])
with col_op:
    op_mode_select = st.radio("운전 모드 선택", ["난방/온수 동시 열 공급", "난방/온수전환 열 공급(3-Way 밸브사용)"])
    final_op_mode = 'priority' if "3-Way" in op_mode_select else 'simultaneous'

def synced_input(label, min_v, max_v, def_v):
    k = f"sl_{label.replace(' ', '_')}"
    if k not in st.session_state: st.session_state[k] = def_v
    st.slider(label, min_v, max_v, key=k)
    return st.session_state[k]

c_eq1, c_eq2 = st.columns(2)
with c_eq1:
    vol_tank = synced_input("물탱크 용량(L)", 100, 600, 300)
    target_temp = synced_input("물탱크 목표온도(℃)", 40, 75, 50)
    hysteresis = synced_input("재가동 온도차(℃)", 2, 20, 5)
with c_eq2:
    target_room_temp = synced_input("실내 목표온도(℃)", 18, 30, 23)
    heating_hours = st.multiselect("난방 가동 시간", list(range(24)), default=[22,23,0,1,2,3,4,5,6], format_func=lambda x: f"{x}시")

if 'dhw_counts' not in st.session_state: st.session_state.dhw_counts = {'shower': 1, 'wash': 1, 'sink': 1, 'laundry': 1}
dhw_schedules = {}
def multi_dhw_ui(label, key_name, def_h, def_m, def_d):
    schedules = []
    with st.expander(f"{label} 설정 ({st.session_state.dhw_counts[key_name]}건)"):
        for i in range(st.session_state.dhw_counts[key_name]):
            c1, c2, c3, c4 = st.columns([0.8, 1.5, 1.5, 1.5])
            with c1: en = st.checkbox("On", True, key=f"{key_name}_{i}_en")
            with c2: h = st.number_input("시", 0, 23, def_h, key=f"{key_name}_{i}_h")
            with c3: m = st.selectbox("분", range(0, 60, 5), index=def_m//5, key=f"{key_name}_{i}_m")
            with c4: d = st.selectbox("분(사용)", range(5, 65, 5), index=(def_d//5)-1, key=f"{key_name}_{i}_d")
            schedules.append({'enabled': en, 'hour': h, 'minute': m, 'duration': d})
        if st.button("추가 +", key=f"btn_{key_name}"): 
            st.session_state.dhw_counts[key_name] += 1
            st.rerun()
    return schedules

col_d1, col_d2, col_d3, col_d4 = st.columns(4)
with col_d1: dhw_schedules['shower'] = multi_dhw_ui("🚿 샤워", "shower", 7, 0, 20)
with col_d2: dhw_schedules['wash'] = multi_dhw_ui("🪥 세면", "wash", 8, 10, 10)
with col_d3: dhw_schedules['sink'] = multi_dhw_ui("🍽️ 설거지", "sink", 19, 30, 20)
with col_d4: dhw_schedules['laundry'] = multi_dhw_ui("👕 세탁기", "laundry", 10, 0, 40)

# (5) 실행
st.markdown("---")
if st.button("🚀 시뮬레이션 및 전기요금 분석 시작", type="primary", use_container_width=True):
    dm = DataManager()
    
    # PV 사용 시 API 자동 조회
    if use_pv:
        with st.spinner("한국전력거래소 SMP/REC API 조회 중..."):
            smp_val, rec_val, smp_ok, rec_ok = dm.fetch_smp_rec_prices(pv_date)
            st.session_state.api_smp = smp_val
            st.session_state.api_rec = rec_val
            st.session_state.smp_ok = smp_ok
            st.session_state.rec_ok = rec_ok

    with st.spinner("기상 데이터 및 NASA 일조량 시뮬레이션 계산 중..."):
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
        'pv_installed': use_pv, 'pv_capacity': pv_cap if use_pv else 0, 'pv_direction': pv_dir if use_pv else '남향'
    }
    
    res = run_simulation(temps, pv_irradiance, dhw_schedules, heating_hours, cfg, start_date)
    tc = TariffCalculator()
    
    cost_res = tc.calculate_cost(tariff_code, res['timestamps'], res['net_power_kwh'], base_monthly_kwh, contract_kw)
    cost_no_pv = tc.calculate_cost(tariff_code, res['timestamps'], res['power_kwh'], base_monthly_kwh, contract_kw)
    pv_savings_krw = cost_no_pv['hp_pure_cost'] - cost_res['hp_pure_cost']
    
    st.session_state.last_res = res
    st.session_state.last_cfg = cfg
    st.session_state.last_cost = cost_res
    st.session_state.pv_savings_krw = pv_savings_krw
    st.session_state.last_start = start_date
    st.rerun()

# (6) 결과 화면
if 'last_res' in st.session_state:
    res = st.session_state.last_res
    cost = st.session_state.last_cost
    cfg = st.session_state.last_cfg
    
    st.header("💰 전기요금 및 경제성 분석 결과 (월간 환산)")
    
    # [알림] API 상태 표시 (장애/지연 시)
    api_warnings = []
    if cfg['pv_installed']:
        if res.get('pv_fallback_used'):
            api_warnings.append("NASA 일조량 API: 데이터 지연/미구축으로 인해 '표준 맑은 날 가상 커브' 자동 적용")
        if not st.session_state.get('smp_ok', False):
            api_warnings.append("한국전력 SMP API: 응답 지연으로 기본 단가(120원) 적용")
        if not st.session_state.get('rec_ok', False):
            api_warnings.append("서부발전 REC API: 응답 지연으로 기본 단가(60원) 적용")
            
    if api_warnings:
        for w in api_warnings:
            st.warning(f"📡 {w}")
    
    days_simulated = max((res['timestamps'][-1] - res['timestamps'][0]).days + 1, 1)
    scale = 30 / days_simulated if days_simulated < 30 else 1.0
    
    if cfg['pv_installed']:
        c_res1, c_res2, c_res3, c_res4 = st.columns(4)
        total_export_month = sum(res['pv_export_kwh']) * scale
        pv_revenue_month = int(total_export_month * (st.session_state.api_smp + st.session_state.api_rec)) 
        real_net_cost = cost['hp_pure_cost'] - pv_revenue_month
        
        with c_res1: st.metric("실제 전기 체감비용", f"{real_net_cost:,} 원", "청구액 - 매전수익", delta_color="inverse")
        with c_res2: st.metric("청구될 히트펌프 요금", f"{cost['hp_pure_cost']:,} 원", f"PV 자가소비 절감액: -{st.session_state.pv_savings_krw:,}원")
        with c_res3: st.metric("월 예상 매전수익", f"{pv_revenue_month:,} 원", f"{total_export_month:.1f} kWh 잉여판매")
        with c_res4: st.metric("적용 단가(SMP/REC)", f"{st.session_state.api_smp}원 / {st.session_state.api_rec}원")
    else:
        c_res1, c_res2, c_res3, c_res4 = st.columns(4)
        with c_res1: st.metric("예상 월간 전기료", f"{cost['total_cost']:,} 원", delta=f"+{cost['hp_pure_cost']:,}원 (HP 단독분)")
        with c_res2: st.metric("히트펌프 월 소비량", f"{sum(res['power_kwh']) * scale:.1f} kWh")
        with c_res3: st.metric("적용 요금제", cost['type'])
        with c_res4: st.metric("1kWh당 평균단가", f"{int(cost['hp_pure_cost'] / (sum(res['power_kwh'])*scale)) if sum(res['power_kwh'])>0 else 0} 원")
        
    st.caption("※ 위 요금은 시뮬레이션 기간의 패턴이 한 달간 지속된다고 가정하여 30일치로 환산한 추정치입니다.")
    st.divider()

    st.subheader("📊 운전 패턴 및 온도 변화")
    
    cold_warnings_dict = analyze_cold_events(res, dhw_schedules, st.session_state.last_start)
    priority_order = ['shower', 'wash', 'sink', 'laundry']
    act_labels = {'shower': '🚿 샤워', 'wash': '🪥 세면', 'sink': '🍽️ 설거지', 'laundry': '👕 세탁기'}
    has_warning = False
    
    for key in priority_order:
        items = cold_warnings_dict.get(key, [])
        if not items: continue
        has_warning = True
        label = act_labels[key]
        
        if len(items) > 1:
            sorted_items = sorted(items, key=lambda x: x['temp'])
            worst = sorted_items[0]
            st.error(f"⚠️ [대표 발생] {worst['msg']} (외 {len(items)-1}건)")
            with st.expander(f"🔻 {label} 온수 부족 전체 내역 보기"):
                for item in sorted_items: st.write(f"- {item['msg']}")
        else:
            st.error(f"⚠️ {items[0]['msg']}")
            
    if not has_warning: st.success("✅ 온수 사용 중 물탱크 온도가 39도 미만으로 떨어진 적이 없습니다.")

    opts = ["난방 가동구간", "온수 가동구간", "찬물(급수) 온도", "히트펌프 소비전력(kW)", "태양광 발전량(kW)"]
    def_opts = ["난방 가동구간", "온수 가동구간"]
    if cfg['pv_installed']: def_opts.append("태양광 발전량(kW)")
    
    plot_options = st.multiselect("그래프에 표시할 항목 선택", options=opts, default=def_opts)

    plot_times = res['timestamps']
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.plot(plot_times, res['tank'], 'b-', label='Tank Temp', lw=1.5)
    ax1.plot(plot_times, res['room'], 'r-', label='Room Temp', lw=2)
    ax1.plot(plot_times, res['amb'], 'g--', label='Outdoor', alpha=0.4)
    if "찬물(급수) 온도" in plot_options: ax1.plot(plot_times, res['inlet'], color='purple', linestyle=':', label='Inlet Water', alpha=0.6)
    
    ymin, ymax = ax1.get_ylim()
    dhw_arr, heat_arr = np.array(res['dhw_active']), np.array(res['heat_active'])
    if "난방 가동구간" in plot_options: ax1.fill_between(plot_times, ymin, ymax, where=heat_arr, color='red', alpha=0.1, label='Heating Active')
    if "온수 가동구간" in plot_options: ax1.fill_between(plot_times, ymin, ymax, where=dhw_arr, color='blue', alpha=0.1, label='DHW Active')

    ax1.set_ylabel("Temperature (C)")
    ax1.legend(loc='upper left', ncol=3)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%Hh'))
    
    if "히트펌프 소비전력(kW)" in plot_options or "태양광 발전량(kW)" in plot_options:
        ax2 = ax1.twinx()
        if "히트펌프 소비전력(kW)" in plot_options:
            ax2.fill_between(plot_times, 0, np.array(res['power_kwh']) * 60, color='orange', alpha=0.3, label='HP Power Input (kW)')
        if "태양광 발전량(kW)" in plot_options and cfg['pv_installed']:
            pv_gen_kw_series = (np.array(res['pv_self_kwh']) + np.array(res['pv_export_kwh'])) * 60
            ax2.plot(plot_times, pv_gen_kw_series, color='green', lw=1.5, alpha=0.7, label='PV Generation (kW)')
        ax2.set_ylabel("Power (kW)")
        ax2.legend(loc='upper right')
    
    st.pyplot(fig)