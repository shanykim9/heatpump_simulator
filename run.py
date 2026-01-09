import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from geopy.geocoders import Nominatim
import requests
from datetime import datetime
import pandas as pd

# ==========================================
# 1. 페이지 설정
# ==========================================
st.set_page_config(page_title="스마트 히트펌프 시뮬레이터", layout="wide")
st.title("🏡 스마트 히트펌프 시뮬레이터 (혼합 밸브 적용)")

# ==========================================
# 2. 핵심 엔진
# ==========================================
class WeatherManager:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="hp_sim_mixing_v1")

    def get_weather_data(self, address, start_date, end_date):
        try:
            location = self.geolocator.geocode(address)
            if not location: return None, "주소를 찾을 수 없습니다."
            
            lat, lon = location.latitude, location.longitude
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": lat, "longitude": lon,
                "start_date": start_date, "end_date": end_date,
                "hourly": "temperature_2m", "timezone": "auto"
            }
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
        return float(cap * 1000), float(cop)

def run_simulation(weather_temps, dhw_schedules, heating_hours, cfg):
    dt = 60 # 1분
    steps = len(weather_temps)
    
    ua = cfg['ua_house']
    heat_cap = cfg['heat_cap_house']
    vol_tank = cfg['vol_tank']
    target_temp = cfg['target_temp']
    hysteresis = cfg['hysteresis']
    
    cp_water = 4186
    temp_tank = 50.0
    temp_room = 20.0
    t_inlet = 5.0 # 직수 온도
    
    res = {'time': [], 'tank': [], 'room': [], 'amb': [], 'cop': []}
    hp_model = RealHeatPumpModel()
    hp_running = False
    
    # [설정] 기기별 사용 유량 (LPM) 및 목표 온도(℃)
    # 목표 온도가 설정되지 않은 항목은 기본적으로 40~45도로 가정하여 계산
    dhw_specs = {
        'shower':  {'flow': 10, 'temp': 40}, # ★ 요청사항 반영 (40도)
        'wash':    {'flow': 6,  'temp': 38}, # ★ 요청사항 반영 (38도)
        'sink':    {'flow': 5,  'temp': 45}, # 설거지는 보통 뜨겁게
        'laundry': {'flow': 12, 'temp': 40}  # 세탁기 온수세탁
    }
    
    for t in range(steps):
        t_amb = weather_temps[t]
        current_hour = (t // 60) % 24
        current_min = t % 60
        
        # [A] 온수 사용량 계산 (혼합 밸브 로직 적용)
        q_dhw = 0
        
        for item, setting in dhw_schedules.items():
            if setting['enabled']:
                s_min = setting['hour'] * 60 + setting['minute']
                e_min = s_min + setting['duration']
                c_total_min = current_hour * 60 + current_min
                
                if s_min <= c_total_min < e_min:
                    spec = dhw_specs[item]
                    target_use_temp = spec['temp'] # 사용자가 원하는 온도 (예: 40도)
                    total_flow_req = spec['flow']  # 수도꼭지에서 나오는 총 유량 (예: 10LPM)
                    
                    # --- 혼합 밸브 계산 ---
                    # 물탱크가 목표온도보다 뜨거우면 찬물을 섞어 씀 -> 탱크 물은 적게 씀
                    # 물탱크가 목표온도보다 차가우면 그냥 탱크 물 100% 씀 (그래도 미지근하겠지만)
                    
                    if temp_tank > target_use_temp:
                        # (사용온도 - 직수온도) / (탱크온도 - 직수온도) 비율만큼만 탱크에서 가져옴
                        hot_ratio = (target_use_temp - t_inlet) / (temp_tank - t_inlet)
                        actual_hot_flow = total_flow_req * hot_ratio
                    else:
                        actual_hot_flow = total_flow_req # 100% 온수 (섞을 필요 없음)
                    
                    # 실제 탱크에서 빠져나간 열량 계산
                    # (실제 빠져나간 온수량) * 비열 * (탱크온도 - 직수온도)
                    q_dhw += (actual_hot_flow / 60) * cp_water * (temp_tank - t_inlet) * dt

        # [B] 난방 부하
        q_heating = 0
        q_loss = ua * (temp_room - t_amb) * dt
        if current_hour in heating_hours and temp_room < 23.0:
            q_heating = 600 * (temp_tank - temp_room) * dt * 0.95
            
        # [C] 방열 손실 (자연 냉각)
        q_tank_loss = 0.5 * (temp_tank - 20.0) * dt
            
        # [D] 히트펌프 제어
        on_temp = target_temp - hysteresis
        if temp_tank < on_temp:
            hp_running = True
        elif temp_tank >= target_temp:
            hp_running = False
            
        q_hp = 0
        cop = 0
        if hp_running:
            cap, cop_val = hp_model.get_performance(t_amb, temp_tank)
            if cap <= 0: hp_running = False
            else:
                q_hp = cap * dt
                cop = cop_val
                
        # [E] 업데이트
        temp_tank += (q_hp - q_heating - q_dhw - q_tank_loss) / (vol_tank * cp_water)
        temp_room += (q_heating - q_loss) / heat_cap
        
        res['time'].append(t/60)
        res['tank'].append(temp_tank)
        res['room'].append(temp_room)
        res['amb'].append(t_amb)
        res['cop'].append(cop)
        
    return res

# ==========================================
# 3. Streamlit UI
# ==========================================
col1, col2, col3 = st.columns(3)
with col1: address = st.text_input("📍 주소 입력", value="Seoul, Gangnam-gu")
with col2: start_date = st.date_input("📅 시작일", datetime(2023, 1, 1))
with col3: end_date = st.date_input("📅 종료일", datetime(2023, 1, 3))

st.divider()

st.subheader("1. 주택 단열 정보")
col_h1, col_h2, col_h3, col_h4, col_h5 = st.columns(5)
with col_h1: year_opt = st.selectbox("건축년도", options=[1.4, 0.7, 0.35], format_func=lambda x: "1980~2000년" if x==1.4 else ("2001~2015년" if x==0.7 else "2016년 이후"))
with col_h2: type_opt = st.selectbox("주택형태", options=['apt', 'house'], format_func=lambda x: "아파트" if x=='apt' else "단독주택")
with col_h3: area_val = st.number_input("전용면적(㎡)", value=85, step=1)
with col_h4: region_opt = st.selectbox("지역", options=[1.1, 1.0, 0.85, 0.7], format_func=lambda x: "중부1" if x==1.1 else ("중부2" if x==1.0 else ("남부" if x==0.85 else "제주")))
with col_h5: pos_opt = st.selectbox("위치", options=['mid', 'top', 'bot'], format_func=lambda x: "중간층" if x=='mid' else ("최상층" if x=='top' else "최하층"))

exposed_area = (np.sqrt(area_val)*4*2.5*0.4) if type_opt == 'apt' else (np.sqrt(area_val)*4*2.5 + area_val*1.5)
if type_opt == 'apt' and pos_opt == 'top': exposed_area += area_val
elif type_opt == 'apt' and pos_opt == 'bot': exposed_area += area_val*0.7
final_ua = year_opt * region_opt * exposed_area
heat_cap = area_val * 200000
st.info(f"👉 난방부하(UA): **{final_ua:.1f} W/K**")

st.divider()

st.subheader("2. 온수 사용 패턴 (혼합 사용 반영)")
st.caption("💡 탱크가 뜨거우면 찬물을 섞어 쓰므로 온수 소모량이 줄어듭니다.")
st.caption("- 샤워: 40℃ 목표 / 세면: 38℃ 목표 / 설거지: 45℃ / 세탁: 40℃")

dhw_schedules = {}
def dhw_ui(label, key_prefix, def_h, def_m, def_d):
    c1, c2, c3, c4 = st.columns([1.5, 2, 2, 2])
    with c1: enabled = st.checkbox(label, value=True, key=f"{key_prefix}_en")
    with c2: hour = st.number_input("시작(시)", 0, 23, def_h, key=f"{key_prefix}_h")
    with c3: minute = st.selectbox("시작(분)", list(range(0, 60, 5)), index=def_m//5, key=f"{key_prefix}_m")
    with c4: duration = st.selectbox("사용시간(분)", list(range(5, 65, 5)), index=(def_d//5)-1, key=f"{key_prefix}_d")
    return {'enabled': enabled, 'hour': hour, 'minute': minute, 'duration': duration}

dhw_schedules['shower'] = dhw_ui("🚿 샤워", "s", 7, 0, 20)
dhw_schedules['wash'] = dhw_ui("🪥 세면", "w", 8, 10, 10)
dhw_schedules['sink'] = dhw_ui("🍽️ 설거지", "k", 19, 30, 20)
dhw_schedules['laundry'] = dhw_ui("👕 세탁기", "l", 10, 0, 40)

st.divider()

st.subheader("3. 설비 및 난방 설정")
col_eq1, col_eq2 = st.columns(2)
with col_eq1:
    vol_tank = st.slider("물탱크 용량(L)", 100, 600, 300)
    target_temp = st.slider("🎯 목표온도(℃)", 40, 75, 55)
    hysteresis = st.slider("🔄 재가동 온도차(℃)", 2, 20, 5)
with col_eq2:
    heating_hours = st.multiselect("🔥 난방 가동 시간", options=list(range(24)), default=[18,19,20,21,22,23,0,1,2,3,4,5,6,7], format_func=lambda x: f"{x:02d}시")

st.divider()

if st.button("▶ 시뮬레이션 실행 (Run)", type="primary", use_container_width=True):
    with st.spinner('데이터 처리 중...'):
        wm = WeatherManager()
        temps, err = wm.get_weather_data(address, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    
    if err: st.error(f"오류: {err}")
    else:
        cfg = {'ua_house': final_ua, 'heat_cap_house': heat_cap, 'vol_tank': vol_tank, 'target_temp': target_temp, 'hysteresis': hysteresis}
        res = run_simulation(temps, dhw_schedules, heating_hours, cfg)
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(res['time'], res['tank'], 'b-', label='Tank Temp', lw=1.5)
        ax1.plot(res['time'], res['room'], 'r-', label='Room Temp', lw=2)
        ax1.plot(res['time'], res['amb'], 'g--', label='Outdoor', alpha=0.4)
        ax1.axhline(y=target_temp, color='orange', ls=':', alpha=0.8, label='Target(OFF)')
        ax1.axhline(y=target_temp - hysteresis, color='purple', ls=':', alpha=0.8, label='Restart(ON)')
        ax1.set_xlabel('Time (Hours)')
        ax1.set_ylabel('Temperature (°C)')
        ax1.legend(loc='upper left', ncol=2)
        ax1.grid(True, alpha=0.3)
        
        ax2 = ax1.twinx()
        ax2.plot(res['time'], res['cop'], 'k:', label='COP', alpha=0.3)
        ax2.set_ylabel('COP')
        ax2.set_ylim(0, 8)
        
        st.pyplot(fig)