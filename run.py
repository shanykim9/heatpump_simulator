"""
=============================================================================
[스마트 히트펌프 에너지 및 요금 시뮬레이터 핵심 로직 요약]

1. 열역학 및 물리 모델 (Thermodynamic & Physical Models)
   - 히트펌프 성능 보간 (2D Interpolation): 외기온도와 출수온도에 따른 난방 능력(Capacity) 및 효율(COP)을 2차원 보간법으로 계산.
   - 제상 로직 (Defrost Derating): 5℃ 미만에서 외기온도에 비례하여 난방 능력이 선형적으로 감소(최대 80% 제한)하는 현실적 제상 손실 반영.
   - 실내 열평형 (Thermal Balance): 주택 단열(UA), 열용량(Heat Capacity), 난방 코일의 열전달(600 W/K)을 이용해 '들어온 열 - 빠져나간 열' 기반의 실내 온도 변화 계산.
   - 물탱크 열손실 (Newton's Cooling): 물탱크 단열 성능(1.84 W/K)과 주위 온도(20℃)의 차이에 비례한 자연 냉각 현상 모사.
   - 직수 온도 (Sinusoidal Model): 계절별 지중/상수도 온도 변화를 연간 사인파(Sine wave) 형태로 추정하여 정밀한 급탕 부하 산출.

2. 제어 및 운영 알고리즘 (Control & Operation Algorithms)
   - 믹싱 밸브 로직 (Mixing Valve): 사용자가 원하는 온수 온도와 찬물(직수) 온도를 혼합비율로 계산하여, 실제 물탱크에서 빠져나가는 뜨거운 물의 유량을 산출.
   - 히스테리시스 제어 (Hysteresis): 잦은 ON/OFF 방지를 위해 설정 온도 대비 지정된 온도차(예: 2~5℃)만큼 떨어졌을 때만 히트펌프가 재가동되도록 제어.
   - 3-Way 밸브 제어 (DHW Priority): 급탕 우선 모드 선택 시, 온수를 사용하는 스케줄 동안에는 난방 공급을 일시 중단하여 온수 온도를 높게 유지하는 로직 적용.

3. 경제성 및 UI/UX 분석 (Economics & UI Analysis)
   - KEPCO 전기요금 산출: 주택용(누진제) 및 일반용/농사용/제주지역(시간대별/계절별 TOU) 요금표를 적용하여 월간 전기요금 예측.
   - 냉수 이벤트 감지 (Cold Water Warning): 시뮬레이션 기간 동안 사용자가 지정한 급탕 스케줄 중 물탱크 온도가 39℃ 미만으로 떨어지는 순간을 포착하여 알림 표시.
=============================================================================
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import RegularGridInterpolator
from geopy.geocoders import Nominatim
import requests
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
# 2. 전기요금 계산기
# ==========================================
class TariffCalculator:
    def __init__(self):
        self.res_low = {'base': [910, 1600, 7300], 'rate': [120, 214.6, 307.3]}
        self.res_high = {'base': [730, 1260, 6060], 'rate': [105, 174, 242.3]}
        self.com_general = {'base_per_kw': 6160, 'summer': 120.0, 'spring_fall': 80.0, 'winter': 105.0}
        self.agri = {'base_per_kw': 360, 'rate': 50.0}
        
    def get_season(self, month):
        if month in [6, 7, 8]: return 'summer'
        elif month in [11, 12, 1, 2]: return 'winter'
        else: return 'spring_fall'

    def get_jeju_tou_rate(self, month, hour):
        season = self.get_season(month)
        is_off_peak = (23 <= hour or hour < 9)
        is_on_peak = (10 <= hour < 12) or (13 <= hour < 17)
        if season == 'summer':
            return 230.0 if is_on_peak else (100.0 if is_off_peak else 150.0)
        elif season == 'winter':
            return 210.0 if is_on_peak else (110.0 if is_off_peak else 145.0)
        else:
            return 140.0 if is_on_peak else (90.0 if is_off_peak else 110.0)

    def calculate_cost(self, tariff_type, location_type, timestamps, powers_kwh, base_monthly_kwh=0, contract_kw=3):
        total_hp_kwh = sum(powers_kwh)
        days = (timestamps[-1] - timestamps[0]).days + 1
        scale_factor = 30 / max(days, 1) if days < 30 else 1.0
        
        estimated_monthly_hp_kwh = total_hp_kwh * scale_factor
        cost_info = {}
        
        if tariff_type.startswith("residential"):
            table = self.res_low if tariff_type == "residential_low" else self.res_high
            def calc_res_bill(kwh):
                if kwh <= 200: 
                    return table['base'][0] + kwh * table['rate'][0]
                elif kwh <= 400:
                    return table['base'][1] + (200 * table['rate'][0]) + ((kwh-200) * table['rate'][1])
                else:
                    return table['base'][2] + (200 * table['rate'][0]) + (200 * table['rate'][1]) + ((kwh-400) * table['rate'][2])

            base_cost = calc_res_bill(base_monthly_kwh)
            final_kwh = base_monthly_kwh + estimated_monthly_hp_kwh
            final_cost = calc_res_bill(final_kwh)
            hp_added_cost = final_cost - base_cost
            
            cost_info = {
                "type": "주택용 누진제",
                "base_kwh": base_monthly_kwh,
                "hp_kwh_month": estimated_monthly_hp_kwh,
                "total_kwh_month": final_kwh,
                "base_cost": int(base_cost),
                "total_cost": int(final_cost),
                "hp_pure_cost": int(hp_added_cost),
                "avg_unit_price": int(hp_added_cost / estimated_monthly_hp_kwh) if estimated_monthly_hp_kwh > 0 else 0
            }
        else:
            basic_charge = self.agri['base_per_kw'] * contract_kw if tariff_type == "agricultural" else self.com_general['base_per_kw'] * contract_kw
            usage_charge_accum = 0
            for t, p_kwh in zip(timestamps, powers_kwh):
                month = t.month
                hour = t.hour
                if location_type == 'jeju': unit_price = self.get_jeju_tou_rate(month, hour)
                elif tariff_type == "agricultural": unit_price = self.agri['rate']
                else: unit_price = self.com_general.get(self.get_season(month), 100)
                usage_charge_accum += p_kwh * unit_price
            
            monthly_usage_charge = usage_charge_accum * scale_factor
            total_month_bill = basic_charge + monthly_usage_charge
            
            cost_info = {
                "type": f"일반/농사 ({'제주 TOU' if location_type=='jeju' else '계절별'})",
                "base_kwh": 0,
                "hp_kwh_month": estimated_monthly_hp_kwh,
                "base_cost": int(basic_charge),
                "total_cost": int(total_month_bill),
                "hp_pure_cost": int(monthly_usage_charge),
                "avg_unit_price": int(monthly_usage_charge / estimated_monthly_hp_kwh) if estimated_monthly_hp_kwh > 0 else 0
            }
            
        return cost_info

# ==========================================
# 3. 핵심 엔진
# ==========================================
class WeatherManager:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="hp_sim_korea_v3")

    def get_weather_data(self, address, start_date, end_date):
        try:
            full_query = f"South Korea, {address}"
            location = self.geolocator.geocode(full_query)
            if not location: return None, f"'{address}' 주소를 찾을 수 없습니다."
            
            lat, lon = location.latitude, location.longitude
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {"latitude": lat, "longitude": lon, "start_date": start_date, "end_date": end_date, "hourly": "temperature_2m", "timezone": "auto"}
            res = requests.get(url, params=params)
            data = res.json()
            if "hourly" not in data: return None, "기상 데이터 수신 실패"
            
            hourly_temps = np.array(data["hourly"]["temperature_2m"])
            x_hours = np.arange(len(hourly_temps)) * 60
            x_minutes = np.arange(len(hourly_temps) * 60)
            return np.interp(x_minutes, x_hours, hourly_temps), None
        except Exception as e:
            return None, str(e)

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
        t_water_safe = min(t_water, 75.0)
        cap = self.interp_cap((t_water_safe, t_amb))
        cop = self.interp_cop((t_water_safe, t_amb))
        if cap is None or cap <= 0.1: return 0.0, 0.0
        
        if t_amb <= -25: defrost_factor = 0.8
        elif t_amb >= 5: defrost_factor = 1.0
        else: defrost_factor = 0.8 + (t_amb + 25) * (0.2 / 30)
        return float(cap * 1000 * defrost_factor), float(cop)

def get_mains_temperature(day_of_year, region_factor=1.0):
    avg_temp = 13.0
    amplitude = 10.0
    if region_factor > 1.05: 
        avg_temp = 10.0
        amplitude = 11.0
    elif region_factor < 0.8: 
        avg_temp = 16.0
        amplitude = 7.0
    rad = (2 * np.pi / 365) * (day_of_year - 35 - (365/4))
    return avg_temp + amplitude * np.sin(rad)

def run_simulation(weather_temps, dhw_schedules, heating_hours, cfg, start_date):
    dt = 60 # 1분 단위
    steps = len(weather_temps)
    ua = cfg['ua_house']
    heat_cap = cfg['heat_cap_house']
    vol_tank = cfg['vol_tank']
    target_temp = cfg['target_temp']
    hysteresis = cfg['hysteresis']
    target_room_temp = cfg.get('target_room_temp', 23.0)
    region_factor = cfg.get('region_factor', 1.0)
    
    op_mode = cfg.get('op_mode', 'simultaneous')
    
    cp_water = 4186
    temp_tank = 50.0
    temp_room = 20.0
    
    res = {'time': [], 'tank': [], 'room': [], 'amb': [], 'cop': [], 'inlet': [], 'dhw_active': [], 'heat_active': [], 'power_kwh': [], 'timestamps': []}
    hp_model = RealHeatPumpModel()
    hp_running = False
    
    dhw_specs = {
        'shower':  {'flow': 10, 'temp': 40},
        'wash':    {'flow': 6,  'temp': 38},
        'sink':    {'flow': 5,  'temp': 45},
        'laundry': {'flow': 12, 'temp': 40}
    }
    
    start_doy = start_date.timetuple().tm_yday
    base_dt = datetime.combine(start_date, datetime.min.time())
    
    for t in range(steps):
        current_dt = base_dt + timedelta(minutes=t)
        t_amb = weather_temps[t]
        current_hour = (t // 60) % 24
        current_min = t % 60
        current_doy = start_doy + (t // (24 * 60))
        t_inlet = get_mains_temperature(current_doy, region_factor)
        
        q_dhw = 0
        dhw_on = False
        for item, schedules in dhw_schedules.items():
            for setting in schedules:
                if setting['enabled']:
                    s_min = setting['hour'] * 60 + setting['minute']
                    e_min = s_min + setting['duration']
                    c_total_min = current_hour * 60 + current_min
                    if s_min <= c_total_min < e_min:
                        dhw_on = True
                        spec = dhw_specs[item]
                        target_use = spec['temp']
                        total_req = spec['flow']
                        if temp_tank > target_use:
                            hot_ratio = (target_use - t_inlet) / (temp_tank - t_inlet)
                            actual_flow = total_req * hot_ratio
                        else:
                            actual_flow = total_req
                        q_dhw += (actual_flow / 60) * cp_water * (temp_tank - t_inlet) * dt

        q_heating = 0
        heat_on = False
        q_loss = ua * (temp_room - t_amb) * dt
        
        if op_mode == 'priority' and dhw_on:
            heat_on = False
            q_heating = 0
        else:
            if current_hour in heating_hours and temp_room < target_room_temp:
                heat_on = True
                q_heating = 600 * (temp_tank - temp_room) * dt * 0.95
            
        q_tank_loss = 1.84 * (temp_tank - 20.0) * dt  # 물탱크 단열성능계수 기존 0.5 -> 1.84
        
        on_temp = target_temp - hysteresis
        if temp_tank < on_temp: hp_running = True
        elif temp_tank >= target_temp: hp_running = False
            
        q_hp = 0
        cop = 0
        input_power_kw = 0
        
        if hp_running:
            cap_w, cop_val = hp_model.get_performance(t_amb, temp_tank)
            if cap_w <= 0: 
                hp_running = False
            else:
                q_hp = cap_w * dt
                cop = cop_val
                if cop_val > 0:
                    input_power_kw = (cap_w / cop_val) / 1000
                
        temp_tank += (q_hp - q_heating - q_dhw - q_tank_loss) / (vol_tank * cp_water)
        temp_room += (q_heating - q_loss) / heat_cap
        
        power_kwh_step = input_power_kw * (1/60)
        
        res['time'].append(t/60)
        res['tank'].append(temp_tank)
        res['room'].append(temp_room)
        res['amb'].append(t_amb)
        res['cop'].append(cop)
        res['inlet'].append(t_inlet)
        res['dhw_active'].append(dhw_on)
        res['heat_active'].append(heat_on)
        res['power_kwh'].append(power_kwh_step)
        res['timestamps'].append(current_dt)
        
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
                    act_name = act_map.get(activity_key, activity_key)
                    msg = f"{time_str} '{act_name}' 중 발생! 온도는 '{min_temp:.1f}도'"
                    warnings[activity_key].append({'time': time_str, 'temp': min_temp, 'msg': msg})
    return warnings

# ==========================================
# 4. Streamlit UI
# ==========================================
st.set_page_config(page_title="스마트 히트펌프 시뮬레이터", layout="wide")
st.title("🏡 스마트 히트펌프 시뮬레이터 (전기요금 분석 포함)")

col1, col2, col3, col4 = st.columns([1.5, 1.5, 1, 1])
with col1:
    selected_sido = st.selectbox("시/도 선택", list(KOREA_LOCATIONS.keys()))
with col2:
    selected_gungu = st.selectbox("시/군/구 선택", KOREA_LOCATIONS.get(selected_sido, []))

full_address = f"{selected_sido} {selected_gungu}"
auto_region_factor = get_climate_region(selected_sido, selected_gungu)

is_jeju = "제주" in selected_sido
location_type_str = "제주도" if is_jeju else "내륙"

with col3: start_date = st.date_input("📅 시작일", datetime(2023, 1, 15))
with col4: end_date = st.date_input("📅 종료일", datetime(2023, 1, 17))

st.divider()

st.subheader(f"1. 전기요금 설정 ({location_type_str} 기준)")
col_t1, col_t2, col_t3 = st.columns(3)

tariff_calculator = TariffCalculator()
base_monthly_kwh = 0
contract_kw = 3
tariff_code = ""

with col_t1:
    if is_jeju:
        st.info("🌴 **제주도**는 일반용(계약전력 3kW) 시간대별 요금제가 적용됩니다.")
        tariff_option = "일반용(제주 TOU)"
        tariff_code = "general_jeju"
    else:
        tariff_option = st.selectbox("요금제 선택", 
                                     ["주택용 (저압)", "주택용 (고압)", "일반용 (갑)I", "농사용 (갑)"], 
                                     index=0)
        if tariff_option == "주택용 (저압)": tariff_code = "residential_low"
        elif tariff_option == "주택용 (고압)": tariff_code = "residential_high"
        elif tariff_option == "일반용 (갑)I": tariff_code = "general_inland"
        else: tariff_code = "agricultural"

with col_t2:
    if "residential" in tariff_code:
        st.markdown("**기존 월간 전기 사용량**")
        base_monthly_kwh = st.selectbox(
            "누진 구간 시작점 (kWh)", 
            [0, 100, 200, 300, 400, 500, 600, 800, 1000],
            index=2,
            help="히트펌프 설치 전, 가정에서 사용하는 월 평균 전기량을 선택하세요."
        )
    else:
        st.markdown("**계약 전력 (kW)**")
        contract_kw = st.number_input("일반/농사용 계약전력", value=3, disabled=True)

with col_t3:
    if "residential" in tariff_code:
        st.success(f"💡 기존 {base_monthly_kwh}kWh 사용 가정.\n\n히트펌프 사용량은 그 **상위 누진구간** 단가로 계산됩니다.")
    elif is_jeju:
        st.warning("⚡ **계절+시간대별(TOU)** 요금 적용\n\n(경부하/중간/최대부하 자동 계산)")
    else:
        st.info("⚡ **계절별** 차등 요금 적용\n\n(시간대별 구분 없음)")

st.divider()

st.subheader("2. 주택 및 설비 정보")
col_h1, col_h2, col_h3, col_h4, col_h5 = st.columns(5)
with col_h1: year_opt = st.selectbox("건축년도", options=[1.4, 0.7, 0.35], format_func=lambda x: "1980~2000년" if x==1.4 else ("2001~2015년" if x==0.7 else "2016년 이후"))
with col_h2: type_opt = st.selectbox("주택형태", options=['apt', 'house'], format_func=lambda x: "아파트" if x=='apt' else "단독주택")
with col_h3: area_val = st.number_input("전용면적(㎡)", value=85, step=1)
with col_h4:
    reg_opts = [1.1, 1.0, 0.85, 0.7]
    idx = reg_opts.index(auto_region_factor) if auto_region_factor in reg_opts else 1
    region_opt = st.selectbox("지역계수(자동)", options=reg_opts, index=idx, format_func=lambda x: "중부1" if x==1.1 else ("중부2" if x==1.0 else ("남부" if x==0.85 else "제주")))
with col_h5: pos_opt = st.selectbox("위치", options=['mid', 'top', 'bot'], format_func=lambda x: "중간층" if x=='mid' else ("최상층" if x=='top' else "최하층"))

exposed = (np.sqrt(area_val)*4*2.5*0.4) if type_opt == 'apt' else (np.sqrt(area_val)*4*2.5 + area_val*1.5)
if type_opt == 'apt' and pos_opt == 'top': exposed += area_val
elif type_opt == 'apt' and pos_opt == 'bot': exposed += area_val*0.7
final_ua = year_opt * region_opt * exposed
heat_cap = area_val * 200000

if 'dhw_counts' not in st.session_state: st.session_state.dhw_counts = {'shower': 1, 'wash': 1, 'sink': 1, 'laundry': 1}
dhw_schedules = {}
def multi_dhw_ui(label, key_name, def_h, def_m, def_d):
    count = st.session_state.dhw_counts[key_name]
    schedules = []
    with st.expander(f"{label} 설정 ({count}건)", expanded=False):
        for i in range(count):
            c1, c2, c3, c4 = st.columns([0.8, 1.5, 1.5, 1.5])
            with c1: enabled = st.checkbox("On", value=True, key=f"{key_name}_{i}_en")
            with c2: hour = st.number_input(f"시(H) {i+1}", 0, 23, def_h, key=f"{key_name}_{i}_h")
            with c3: minute = st.selectbox(f"분(M) {i+1}", list(range(0, 60, 5)), index=def_m//5, key=f"{key_name}_{i}_m")
            with c4: duration = st.selectbox(f"시간(Min) {i+1}", list(range(5, 65, 5)), index=(def_d//5)-1, key=f"{key_name}_{i}_d")
            schedules.append({'enabled': enabled, 'hour': hour, 'minute': minute, 'duration': duration})
        if st.button(f"추가 +", key=f"btn_{key_name}"):
            st.session_state.dhw_counts[key_name] += 1
            st.rerun()
    return schedules

col_d1, col_d2, col_d3, col_d4 = st.columns(4)
with col_d1: dhw_schedules['shower'] = multi_dhw_ui("🚿 샤워", "shower", 7, 0, 20)
with col_d2: dhw_schedules['wash'] = multi_dhw_ui("🪥 세면", "wash", 8, 10, 10)
with col_d3: dhw_schedules['sink'] = multi_dhw_ui("🍽️ 설거지", "sink", 19, 30, 20)
with col_d4: dhw_schedules['laundry'] = multi_dhw_ui("👕 세탁기", "laundry", 10, 0, 40)

def _make_safe_key(label): return label.replace(" ", "_")
def synced_input(label, min_v, max_v, def_v):
    sl_key, num_key = f"sl_{_make_safe_key(label)}", f"num_{_make_safe_key(label)}"
    if sl_key not in st.session_state: st.session_state[sl_key] = def_v
    if num_key not in st.session_state: st.session_state[num_key] = def_v
    def up_num(): st.session_state[num_key] = st.session_state[sl_key]
    def up_sl(): st.session_state[sl_key] = st.session_state[num_key]
    c1, c2 = st.columns([3, 1])
    c1.slider(label, min_v, max_v, key=sl_key, on_change=up_num)
    c2.number_input("", min_v, max_v, key=num_key, on_change=up_sl, label_visibility="collapsed")
    return st.session_state[sl_key]

st.markdown("---")
st.subheader("3. 히트펌프 운전 모드 및 제어")
col_op, col_none = st.columns([2, 1])
with col_op:
    op_mode_select = st.radio(
        "운전 모드 선택",
        options=["난방/온수 동시 열 공급", "난방/온수전환 열 공급(3-Way 밸브사용)"],
        index=0,
        help="3-Way 밸브 모드는 온수 사용 시 난방을 잠시 중단합니다 (급탕 우선)."
    )
    final_op_mode = 'priority' if "3-Way" in op_mode_select else 'simultaneous'

st.write("")
c_eq1, c_eq2 = st.columns(2)
with c_eq1:
    vol_tank = synced_input("물탱크 용량(L)", 100, 600, 300)
    target_temp = synced_input("물탱크 목표온도(℃)", 40, 75, 50)
    hysteresis = synced_input("🔄 재가동 온도차(℃)", 2, 20, 5)
with c_eq2:
    target_room_temp = synced_input("실내 목표온도(℃)", 18, 30, 23)
    heating_hours = st.multiselect("난방 가동 시간", list(range(24)), default=[22,23,0,1,2,3,4,5,6], format_func=lambda x: f"{x}시")

st.markdown("---")
if st.button("🚀 시뮬레이션 및 전기요금 분석 시작", type="primary", use_container_width=True):
    with st.spinner("날씨 데이터 수집 및 에너지 시뮬레이션 중..."):
        wm = WeatherManager()
        temps, err = wm.get_weather_data(full_address, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    
    if err: st.error(err)
    else:
        cfg = {
            'ua_house': final_ua, 
            'heat_cap_house': heat_cap, 
            'vol_tank': vol_tank, 
            'target_temp': target_temp, 
            'hysteresis': hysteresis, 
            'target_room_temp': target_room_temp, 
            'region_factor': region_opt,
            'op_mode': final_op_mode
        }
        
        res = run_simulation(temps, dhw_schedules, heating_hours, cfg, start_date)
        loc_type = "jeju" if is_jeju else "inland"
        cost_res = tariff_calculator.calculate_cost(tariff_code, loc_type, res['timestamps'], res['power_kwh'], base_monthly_kwh, contract_kw)
        
        st.session_state.last_res = res
        st.session_state.last_cfg = cfg
        st.session_state.last_cost = cost_res
        st.session_state.last_start = start_date

if 'last_res' in st.session_state:
    res = st.session_state.last_res
    cost = st.session_state.last_cost
    
    st.header("💰 전기요금 분석 결과 (월간 환산)")
    c_res1, c_res2, c_res3, c_res4 = st.columns(4)
    with c_res1: st.metric("예상 월간 전기료", f"{cost['total_cost']:,} 원", delta=f"+{cost['hp_pure_cost']:,}원 (HP증가분)")
    with c_res2: st.metric("히트펌프 월 소비량", f"{cost['hp_kwh_month']:.1f} kWh", f"평균 COP {np.mean([c for c in res['cop'] if c > 0]):.2f}")
    with c_res3: st.metric("적용 요금제", cost['type'])
    with c_res4: st.metric("1kWh당 평균단가", f"{cost['avg_unit_price']} 원", "구간/시간 가중평균")
    st.caption("※ 위 요금은 시뮬레이션 기간의 패턴이 한 달간 지속된다고 가정하여 30일치로 환산한 추정치입니다. (전력기금/부가세 제외 순수요금)")
    
    st.divider()

    st.subheader("📊 운전 패턴 및 온도 변화")
    
    # [수정 완료] 온수 부족 경고 표시 로직 개선
    cold_warnings_dict = analyze_cold_events(res, dhw_schedules, st.session_state.last_start)
    
    # 1. 고정된 우선순위: 샤워 -> 세면 -> 설거지 -> 세탁기
    priority_order = ['shower', 'wash', 'sink', 'laundry']
    act_labels = {'shower': '🚿 샤워', 'wash': '🪥 세면', 'sink': '🍽️ 설거지', 'laundry': '👕 세탁기'}
    
    has_warning = False
    
    for key in priority_order:
        items = cold_warnings_dict.get(key, [])
        if not items: continue
        
        has_warning = True
        label = act_labels[key]
        
        # [Case A] 발생 건수 > 1건: 대표 표시 + 더보기 버튼
        if len(items) > 1:
            sorted_items = sorted(items, key=lambda x: x['temp'])
            worst = sorted_items[0]
            st.error(f"⚠️ [대표 발생] {worst['msg']} (외 {len(items)-1}건)")
            
            with st.expander(f"🔻 {label} 온수 부족 전체 내역 보기"):
                for item in sorted_items:
                    st.write(f"- {item['msg']}")

        # [Case B] 발생 건수 == 1건: 그냥 표시 (더보기 없음)
        else:
            item = items[0]
            st.error(f"⚠️ {item['msg']}")
    
    if not has_warning:
        st.success("✅ 온수 사용 중 물탱크 온도가 39도 미만으로 떨어진 적이 없습니다.")

    plot_options = st.multiselect(
        "그래프에 표시할 항목 선택",
        options=["난방 가동구간", "온수 가동구간", "찬물(급수) 온도", "소비전력(kW)"],
        default=["난방 가동구간", "온수 가동구간", "찬물(급수) 온도"] 
    )

    base_dt = datetime.combine(st.session_state.last_start, datetime.min.time())
    plot_times = [base_dt + timedelta(minutes=t) for t in range(len(res['time']))]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.plot(plot_times, res['tank'], 'b-', label='Tank Temp', lw=1.5)
    ax1.plot(plot_times, res['room'], 'r-', label='Room Temp', lw=2)
    ax1.plot(plot_times, res['amb'], 'g--', label='Outdoor', alpha=0.4)
    
    if "찬물(급수) 온도" in plot_options:
        ax1.plot(plot_times, res['inlet'], color='purple', linestyle=':', label='Inlet Water', alpha=0.6, lw=1.5)
    
    ymin, ymax = ax1.get_ylim()
    dhw_arr = np.array(res['dhw_active'])
    heat_arr = np.array(res['heat_active'])
    
    if "난방 가동구간" in plot_options:
        ax1.fill_between(plot_times, ymin, ymax, where=heat_arr, color='red', alpha=0.1, label='Heating Active')
    
    if "온수 가동구간" in plot_options:
        ax1.fill_between(plot_times, ymin, ymax, where=dhw_arr, color='blue', alpha=0.1, label='DHW Active')

    ax1.set_ylabel("Temperature (C)")
    ax1.legend(loc='upper left', ncol=3)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%Hh'))
    
    if "소비전력(kW)" in plot_options:
        ax2 = ax1.twinx()
        kw_series = np.array(res['power_kwh']) * 60 
        ax2.fill_between(plot_times, 0, kw_series, color='orange', alpha=0.3, label='Power Input (kW)')
        ax2.set_ylabel("Power Input (kW)")
        ax2.legend(loc='upper right')
        ax2.axis('on') 
    
    st.pyplot(fig)