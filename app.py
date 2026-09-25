import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="AgriBridge", page_icon="🌾", layout="wide")

# ---------------------------------------------------------------------------
# Static data
# ---------------------------------------------------------------------------
DISEASES = [
    {"name": "Healthy", "severity": "ok",
     "tip": "No action needed — maintain regular watering and monitor weekly."},
    {"name": "Leaf Blight", "severity": "bad",
     "tip": "Remove infected leaves, apply copper-based fungicide, improve field drainage."},
    {"name": "Powdery Mildew", "severity": "warn",
     "tip": "Increase airflow between plants, apply sulfur spray, avoid overhead watering."},
    {"name": "Bacterial Leaf Spot", "severity": "bad",
     "tip": "Use disease-free seed, rotate crops, apply copper bactericide early morning."},
    {"name": "Leaf Rust", "severity": "warn",
     "tip": "Apply triazole fungicide, remove volunteer plants, choose resistant varieties next season."},
    {"name": "Nutrient Deficiency (Nitrogen)", "severity": "warn",
     "tip": "Apply nitrogen-rich fertilizer or compost; soil test recommended."},
]
SEVERITY_COLOR = {"ok": "green", "warn": "orange", "bad": "red"}
STEPS = ["Order Placed", "Farmer Confirmed", "In Transit", "Delivered"]

CROP_RULES = [
    ("Kharif (Monsoon)", "Clay", ["Rice", "Jute", "Sugarcane", "Cotton"]),
    ("Kharif (Monsoon)", "Loamy", ["Maize", "Soybean", "Groundnut", "Cotton"]),
    ("Kharif (Monsoon)", "Sandy", ["Bajra (Pearl Millet)", "Groundnut", "Watermelon"]),
    ("Rabi (Winter)", "Clay", ["Wheat", "Gram (Chickpea)", "Mustard"]),
    ("Rabi (Winter)", "Loamy", ["Wheat", "Barley", "Peas", "Mustard"]),
    ("Rabi (Winter)", "Sandy", ["Mustard", "Gram (Chickpea)", "Barley"]),
    ("Zaid (Summer)", "Clay", ["Sugarcane", "Vegetables (Okra, Gourd)"]),
    ("Zaid (Summer)", "Loamy", ["Watermelon", "Muskmelon", "Cucumber"]),
    ("Zaid (Summer)", "Sandy", ["Watermelon", "Bajra (Pearl Millet)", "Fodder crops"]),
]

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_state():
    defaults = {
        "listings": [
            {"crop": "Tomatoes", "qty": "120 kg", "price": "₹28/kg", "farmer": "R. Kumar", "loc": "Erode, TN"},
            {"crop": "Basmati Rice", "qty": "400 kg", "price": "₹65/kg", "farmer": "S. Patel", "loc": "Karnal, HR"},
            {"crop": "Onions", "qty": "250 kg", "price": "₹22/kg", "farmer": "M. Reddy", "loc": "Nashik, MH"},
        ],
        "orders": [
            {"item": "Tomatoes (40kg)", "buyer": "Green Mart", "status": 2, "placed": datetime.now() - timedelta(days=2)},
            {"item": "Rice (100kg)", "buyer": "Local Co-op", "status": 1, "placed": datetime.now() - timedelta(days=1)},
        ],
        "scans": [],
        "posts": [
            {"author": "S. Patel", "text": "Anyone dealing with whitefly on cotton this season? Neem oil worked for me.", "time": datetime.now() - timedelta(hours=5)},
            {"author": "M. Reddy", "text": "Onion prices looking good in Nashik mandi this week — ₹22-24/kg.", "time": datetime.now() - timedelta(hours=2)},
        ],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def analyze_leaf(image: Image.Image):
    img = image.convert("RGB").resize((80, 80))
    arr = np.array(img).astype(float)
    r, g, b = arr[:, :, 0].mean(), arr[:, :, 1].mean(), arr[:, :, 2].mean()
    greenness = g - (r + b) / 2
    brown_yellow = (r + g) / 2 - b
    if greenness > 25:
        idx, confidence = 0, 88
    elif brown_yellow > 40 and r > g:
        idx, confidence = 1, 79
    elif g > r and g > b:
        idx, confidence = 2, 74
    elif r > 100 and g < 90:
        idx, confidence = 3, 71
    else:
        idx, confidence = 4, 68
    d = DISEASES[idx]
    return {"name": d["name"], "severity": d["severity"], "tip": d["tip"], "confidence": confidence}


@st.cache_data(ttl=1800, show_spinner=False)
def get_weather(location: str):
    """Fetch live forecast via Open-Meteo (no API key required). Returns None on failure."""
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1}, timeout=6
        ).json()
        if not geo.get("results"):
            return None
        place = geo["results"][0]
        lat, lon = place["latitude"], place["longitude"]
        wx = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "auto", "forecast_days": 5,
            }, timeout=6
        ).json()
        return {
            "place": f"{place.get('name')}, {place.get('country', '')}",
            "current": wx.get("current", {}),
            "daily": wx.get("daily", {}),
        }
    except Exception:
        return None


def emi_calculator(principal, annual_rate, months):
    r = annual_rate / 12 / 100
    if r == 0:
        return principal / months
    return principal * r * (1 + r) ** months / ((1 + r) ** months - 1)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 style='margin-bottom:0;'>🌾 AgriBridge</h1>"
    "<p style='color:gray;margin-top:0;'>Connecting farmers to markets, diagnostics, weather, credit and community</p>",
    unsafe_allow_html=True,
)

tabs = st.tabs([
    "📊 Dashboard", "🔬 Disease Scanner", "🛒 Marketplace", "🚚 Logistics",
    "🌦️ Weather", "🌱 Crop Advisory", "💰 Finance", "💬 Community",
])
(tab_dash, tab_scan, tab_market, tab_logistics,
 tab_weather, tab_advisory, tab_finance, tab_community) = tabs

# --- Dashboard ---
with tab_dash:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active listings", len(st.session_state.listings))
    c2.metric("Orders in logistics", len(st.session_state.orders))
    c3.metric("Crop scans performed", len(st.session_state.scans))
    c4.metric("Community posts", len(st.session_state.posts))

    st.subheader("Recent orders")
    if st.session_state.orders:
        for o in st.session_state.orders:
            st.write(f"**{o['item']}** — Buyer: {o['buyer']} — Status: *{STEPS[o['status']]}*")
    else:
        st.caption("No orders yet")

    st.subheader("Market price trends (per kg)")
    price_df = pd.DataFrame({
        "Day": pd.date_range(end=datetime.now(), periods=7).strftime("%a"),
        "Tomatoes": [24, 25, 26, 27, 26, 28, 28],
        "Rice": [64, 64, 65, 65, 66, 65, 65],
        "Onions": [25, 24, 23, 23, 22, 22, 22],
    }).set_index("Day")
    st.line_chart(price_df)

# --- Disease Scanner ---
with tab_scan:
    st.subheader("Crop Disease Scanner")
    uploaded = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        image = Image.open(uploaded)
        col1, col2 = st.columns([1, 1.4])
        with col1:
            st.image(image, use_container_width=True)
        with col2:
            result = analyze_leaf(image)
            st.session_state.scans.append(result)
            st.markdown(f":{SEVERITY_COLOR[result['severity']]}[**{result['name']}**]")
            st.write(result["tip"])
            st.caption(f"Confidence: {result['confidence']}% · demo heuristic based on leaf color analysis")

    st.subheader("Scan history")
    if st.session_state.scans:
        for s in reversed(st.session_state.scans):
            with st.container(border=True):
                st.markdown(f":{SEVERITY_COLOR[s['severity']]}[**{s['name']}**]  ·  {s['confidence']}% confidence")
                st.caption(s["tip"])
    else:
        st.caption("No scans yet — upload a photo above")

# --- Marketplace ---
with tab_market:
    st.subheader("List your produce")
    with st.form("add_listing", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns(4)
        crop = c1.text_input("Crop", placeholder="e.g. Wheat")
        qty = c2.text_input("Quantity", placeholder="e.g. 100 kg")
        price = c3.text_input("Price", placeholder="e.g. ₹30/kg")
        loc = c4.text_input("Location", placeholder="e.g. Salem, TN")
        if st.form_submit_button("+ Add Listing", type="primary"):
            if crop and qty and price:
                st.session_state.listings.insert(
                    0, {"crop": crop, "qty": qty, "price": price, "farmer": "You", "loc": loc or "—"}
                )
                st.success(f"Listed {crop}")
            else:
                st.error("Please fill crop, quantity and price")

    st.subheader("Available produce")
    fc1, fc2, fc3 = st.columns([2, 2, 1])
    search = fc1.text_input("🔍 Search by crop", placeholder="e.g. rice")
    loc_filter = fc2.text_input("📍 Filter by location", placeholder="e.g. TN")
    filtered = [
        l for l in st.session_state.listings
        if search.lower() in l["crop"].lower()
        and loc_filter.lower() in l["loc"].lower()
    ]
    csv = pd.DataFrame(st.session_state.listings).to_csv(index=False).encode("utf-8")
    fc3.download_button("⬇️ Export CSV", csv, "agribridge_listings.csv", "text/csv")

    cols = st.columns(3)
    for i, l in enumerate(filtered):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"**{l['crop']}**")
                st.caption(f"{l['qty']} · {l['price']}")
                st.caption(f"{l['farmer']} · {l['loc']}")
                if st.button("Buy / Contact", key=f"buy_{i}_{l['crop']}"):
                    st.session_state.orders.insert(
                        0, {"item": f"{l['crop']} ({l['qty']})", "buyer": "You", "status": 0, "placed": datetime.now()}
                    )
                    st.success("Order placed — check the Logistics tab")
    if not filtered:
        st.caption("No listings match your search/filter")

# --- Logistics ---
with tab_logistics:
    st.subheader("Active orders")
    if not st.session_state.orders:
        st.caption("No orders yet — buy something in the Marketplace")
    for idx, o in enumerate(st.session_state.orders):
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"**{o['item']}**")
            eta = o["placed"] + timedelta(days=4 - o["status"])
            c1.caption(f"Buyer: {o['buyer']} · Est. delivery: {eta.strftime('%d %b')}")
            if o["status"] < 3:
                if c2.button("Advance status", key=f"adv_{idx}"):
                    o["status"] += 1
            else:
                c2.success("Complete")
            st.progress(o["status"] / (len(STEPS) - 1), text=STEPS[o["status"]])

# --- Weather ---
with tab_weather:
    st.subheader("Weather forecast")
    loc = st.text_input("Enter your location", value="Chennai, India")
    if st.button("Get forecast") or loc:
        data = get_weather(loc)
        if data is None:
            st.warning("Couldn't fetch live weather (check location spelling or your internet connection).")
        else:
            cur = data["current"]
            st.markdown(f"**{data['place']}**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Temperature", f"{cur.get('temperature_2m', '—')}°C")
            c2.metric("Humidity", f"{cur.get('relative_humidity_2m', '—')}%")
            c3.metric("Wind", f"{cur.get('wind_speed_10m', '—')} km/h")
            c4.metric("Precipitation", f"{cur.get('precipitation', '—')} mm")

            daily = data["daily"]
            if daily:
                df = pd.DataFrame({
                    "Date": daily.get("time", []),
                    "Max °C": daily.get("temperature_2m_max", []),
                    "Min °C": daily.get("temperature_2m_min", []),
                    "Rain chance %": daily.get("precipitation_probability_max", []),
                }).set_index("Date")
                st.write("5-day outlook")
                st.dataframe(df, use_container_width=True)
                st.line_chart(df[["Max °C", "Min °C"]])

# --- Crop Advisory ---
with tab_advisory:
    st.subheader("Crop recommendation")
    c1, c2 = st.columns(2)
    season = c1.selectbox("Season", ["Kharif (Monsoon)", "Rabi (Winter)", "Zaid (Summer)"])
    soil = c2.selectbox("Soil type", ["Clay", "Loamy", "Sandy"])
    if st.button("Get recommendations", type="primary"):
        match = next((c for s, so, c in CROP_RULES if s == season and so == soil), [])
        if match:
            st.success(f"Recommended crops for **{season}** on **{soil} soil**:")
            for crop in match:
                st.write(f"🌾 {crop}")
        else:
            st.info("No specific match found — consult your local agricultural extension office.")

# --- Finance ---
with tab_finance:
    st.subheader("Farm loan / EMI calculator")
    c1, c2, c3 = st.columns(3)
    principal = c1.number_input("Loan amount (₹)", min_value=1000, value=50000, step=1000)
    rate = c2.number_input("Annual interest rate (%)", min_value=1.0, value=7.0, step=0.5)
    months = c3.number_input("Tenure (months)", min_value=1, value=12, step=1)
    emi = emi_calculator(principal, rate, months)
    st.metric("Estimated monthly EMI", f"₹{emi:,.0f}")
    st.caption(f"Total repayment: ₹{emi*months:,.0f} · Total interest: ₹{(emi*months - principal):,.0f}")
    st.caption("Illustrative estimate only — actual rates depend on the lender and government agri-credit schemes (e.g. Kisan Credit Card).")

# --- Community ---
with tab_community:
    st.subheader("Farmer community board")
    with st.form("new_post", clear_on_submit=True):
        name = st.text_input("Your name", placeholder="e.g. A. Farmer")
        text = st.text_area("Share a question, tip or update", placeholder="What's on your mind?")
        if st.form_submit_button("Post"):
            if text.strip():
                st.session_state.posts.insert(0, {"author": name or "Anonymous", "text": text, "time": datetime.now()})
                st.success("Posted!")

    for p in st.session_state.posts:
        with st.container(border=True):
            st.markdown(f"**{p['author']}** · {p['time'].strftime('%d %b, %I:%M %p')}")
            st.write(p["text"])

st.markdown("---")
st.caption("Built for the Agri Bridge hackathon · prototype demo")
