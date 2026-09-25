import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import requests
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from datetime import datetime, timedelta

st.set_page_config(page_title="AgriBridge", page_icon="🌾", layout="wide")

# ---------------------------------------------------------------------------
# Theme / background styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=Outfit:wght@600;700&display=swap');

html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }

/* Page background: soft cream-to-green wash with a faint leaf pattern */
[data-testid="stAppViewContainer"] {
    background:
      radial-gradient(circle at 15% 20%, rgba(39,103,73,0.06) 0, transparent 35%),
      radial-gradient(circle at 85% 75%, rgba(217,119,6,0.07) 0, transparent 40%),
      linear-gradient(180deg, #faf7ef 0%, #f3f6ec 100%);
}
[data-testid="stHeader"] { background: rgba(0,0,0,0); }

/* Hero banner */
.agri-banner{
    background: linear-gradient(120deg, #1b4a34 0%, #276749 55%, #3d7a52 100%);
    border-radius: 18px;
    padding: 22px 28px;
    margin-bottom: 6px;
    box-shadow: 0 8px 24px rgba(27,74,52,0.25);
}
.agri-banner h1{
    font-family:'Outfit', sans-serif; color:#fff; margin:0; font-size:2rem;
}
.agri-banner p{ color:#e6f0e6; margin:4px 0 0; font-size:0.95rem; }

/* Tabs */
.stTabs [data-baseweb="tab-list"]{ gap:4px; }
.stTabs [data-baseweb="tab"]{
    background:#ffffffaa; border-radius:10px 10px 0 0; padding:8px 14px; font-weight:600;
}
.stTabs [aria-selected="true"]{ background:#276749 !important; color:#fff !important; }

/* Buttons */
.stButton>button, .stFormSubmitButton>button, .stDownloadButton>button{
    border-radius:10px; border:1px solid #d9d2bd; font-weight:700;
}
.stButton>button[kind="primary"], .stFormSubmitButton>button[kind="primary"]{
    background:#d97706; border-color:#d97706;
}
.stButton>button[kind="primary"]:hover{ background:#b45309; border-color:#b45309; }

/* Metric + bordered containers -> card look */
div[data-testid="stMetric"], div[data-testid="stVerticalBlockBorderWrapper"]{
    background:#ffffff; border-radius:14px; padding:6px 10px;
    box-shadow: 0 2px 10px rgba(27,74,52,0.08); border:1px solid #ece6d6;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# i18n — English / Hindi / Tamil
# ---------------------------------------------------------------------------
LANGS = {"English": "en", "हिन्दी": "hi", "தமிழ்": "ta"}

T = {
    "en": {
        "subtitle": "Connecting farmers to markets, diagnostics, weather, credit and community",
        "tab_dash": "📊 Dashboard", "tab_scan": "🔬 Disease Scanner", "tab_market": "🛒 Marketplace",
        "tab_logistics": "🚚 Logistics", "tab_weather": "🌦️ Weather", "tab_advisory": "🌱 Crop Advisory",
        "tab_finance": "💰 Finance", "tab_community": "💬 Community",
        "active_listings": "Active listings", "orders_logistics": "Orders in logistics",
        "scans_performed": "Crop scans performed", "community_posts": "Community posts",
        "recent_orders": "Recent orders", "no_orders": "No orders yet",
        "price_trends": "Market price trends (per kg)", "buyer": "Buyer", "status": "Status",
        "scan_title": "Crop Disease Scanner", "upload_leaf": "Upload a leaf photo",
        "running_inference": "Running inference...", "top3": "See top-3 model predictions",
        "confidence": "Confidence",
        "model_note": "Model: MobileNetV2 fine-tuned on the PlantVillage dataset (38 classes, ~95% eval accuracy). Not a substitute for expert diagnosis.",
        "scan_history": "Scan history", "no_scans": "No scans yet — upload a photo above",
        "list_produce": "List your produce", "crop": "Crop", "quantity": "Quantity", "price": "Price",
        "location": "Location", "add_listing": "+ Add Listing",
        "fill_error": "Please fill crop, quantity and price", "listed_success": "Listed {crop}",
        "available_produce": "Available produce", "search_crop": "🔍 Search by crop",
        "filter_loc": "📍 Filter by location", "export_csv": "⬇️ Export CSV",
        "buy_contact": "Buy / Contact", "order_success": "Order placed — check the Logistics tab",
        "no_match": "No listings match your search/filter",
        "active_orders": "Active orders", "no_orders_market": "No orders yet — buy something in the Marketplace",
        "est_delivery": "Est. delivery", "advance_status": "Advance status", "complete": "Complete",
        "weather_title": "Weather forecast", "enter_location": "Enter your location",
        "get_forecast": "Get forecast",
        "weather_fail": "Couldn't fetch live weather (check location spelling or your internet connection).",
        "temperature": "Temperature", "humidity": "Humidity", "wind": "Wind", "precipitation": "Precipitation",
        "outlook_5day": "5-day outlook",
        "advisory_title": "Crop recommendation", "season": "Season", "soil": "Soil type",
        "get_recommendations": "Get recommendations", "recommended_for": "Recommended crops for **{season}** on **{soil} soil**:",
        "no_advisory_match": "No specific match found — consult your local agricultural extension office.",
        "finance_title": "Farm loan / EMI calculator", "loan_amount": "Loan amount (₹)",
        "interest_rate": "Annual interest rate (%)", "tenure": "Tenure (months)",
        "est_emi": "Estimated monthly EMI",
        "repayment_caption": "Total repayment: ₹{total:,.0f} · Total interest: ₹{interest:,.0f}",
        "finance_disclaimer": "Illustrative estimate only — actual rates depend on the lender and government agri-credit schemes (e.g. Kisan Credit Card).",
        "community_title": "Farmer community board", "your_name": "Your name",
        "share_placeholder": "Share a question, tip or update", "post_btn": "Post", "posted": "Posted!",
        "footer": "Built for the Agri Bridge hackathon · prototype demo",
        "language": "Language / भाषा / மொழி",
    },
    "hi": {
        "subtitle": "किसानों को बाज़ार, रोग निदान, मौसम, ऋण और समुदाय से जोड़ना",
        "tab_dash": "📊 डैशबोर्ड", "tab_scan": "🔬 रोग स्कैनर", "tab_market": "🛒 बाज़ार",
        "tab_logistics": "🚚 रसद", "tab_weather": "🌦️ मौसम", "tab_advisory": "🌱 फसल सलाह",
        "tab_finance": "💰 वित्त", "tab_community": "💬 समुदाय",
        "active_listings": "सक्रिय लिस्टिंग", "orders_logistics": "रसद में ऑर्डर",
        "scans_performed": "किए गए फसल स्कैन", "community_posts": "समुदाय पोस्ट",
        "recent_orders": "हाल के ऑर्डर", "no_orders": "अभी तक कोई ऑर्डर नहीं",
        "price_trends": "बाज़ार मूल्य रुझान (प्रति किग्रा)", "buyer": "खरीदार", "status": "स्थिति",
        "scan_title": "फसल रोग स्कैनर", "upload_leaf": "पत्ती की फोटो अपलोड करें",
        "running_inference": "विश्लेषण चल रहा है...", "top3": "शीर्ष-3 मॉडल भविष्यवाणियां देखें",
        "confidence": "विश्वास स्तर",
        "model_note": "मॉडल: प्लांटविलेज डेटासेट पर प्रशिक्षित MobileNetV2 (38 वर्ग, ~95% सटीकता)। यह विशेषज्ञ निदान का विकल्प नहीं है।",
        "scan_history": "स्कैन इतिहास", "no_scans": "अभी तक कोई स्कैन नहीं — ऊपर एक फोटो अपलोड करें",
        "list_produce": "अपनी उपज सूचीबद्ध करें", "crop": "फसल", "quantity": "मात्रा", "price": "मूल्य",
        "location": "स्थान", "add_listing": "+ लिस्टिंग जोड़ें",
        "fill_error": "कृपया फसल, मात्रा और मूल्य भरें", "listed_success": "{crop} सूचीबद्ध किया गया",
        "available_produce": "उपलब्ध उपज", "search_crop": "🔍 फसल से खोजें",
        "filter_loc": "📍 स्थान से फ़िल्टर करें", "export_csv": "⬇️ CSV निर्यात करें",
        "buy_contact": "खरीदें / संपर्क करें", "order_success": "ऑर्डर दिया गया — रसद टैब देखें",
        "no_match": "आपकी खोज/फ़िल्टर से कोई लिस्टिंग मेल नहीं खाती",
        "active_orders": "सक्रिय ऑर्डर", "no_orders_market": "अभी तक कोई ऑर्डर नहीं — बाज़ार में कुछ खरीदें",
        "est_delivery": "अनुमानित डिलीवरी", "advance_status": "स्थिति आगे बढ़ाएं", "complete": "पूर्ण",
        "weather_title": "मौसम पूर्वानुमान", "enter_location": "अपना स्थान दर्ज करें",
        "get_forecast": "पूर्वानुमान प्राप्त करें",
        "weather_fail": "लाइव मौसम प्राप्त नहीं हो सका (स्थान की स्पेलिंग या इंटरनेट कनेक्शन जांचें)।",
        "temperature": "तापमान", "humidity": "आर्द्रता", "wind": "हवा", "precipitation": "वर्षा",
        "outlook_5day": "5-दिन का पूर्वानुमान",
        "advisory_title": "फसल सिफारिश", "season": "मौसम/सीज़न", "soil": "मिट्टी का प्रकार",
        "get_recommendations": "सिफारिशें प्राप्त करें", "recommended_for": "**{season}** मौसम और **{soil}** मिट्टी के लिए अनुशंसित फसलें:",
        "no_advisory_match": "कोई विशेष मेल नहीं मिला — अपने स्थानीय कृषि विस्तार कार्यालय से सलाह लें।",
        "finance_title": "कृषि ऋण / EMI कैलकुलेटर", "loan_amount": "ऋण राशि (₹)",
        "interest_rate": "वार्षिक ब्याज दर (%)", "tenure": "अवधि (महीने)",
        "est_emi": "अनुमानित मासिक EMI",
        "repayment_caption": "कुल पुनर्भुगतान: ₹{total:,.0f} · कुल ब्याज: ₹{interest:,.0f}",
        "finance_disclaimer": "केवल अनुमानित — वास्तविक दरें ऋणदाता और सरकारी कृषि-ऋण योजनाओं (जैसे किसान क्रेडिट कार्ड) पर निर्भर करती हैं।",
        "community_title": "किसान समुदाय बोर्ड", "your_name": "आपका नाम",
        "share_placeholder": "एक प्रश्न, सुझाव या अपडेट साझा करें", "post_btn": "पोस्ट करें", "posted": "पोस्ट हो गया!",
        "footer": "एग्री ब्रिज हैकथॉन के लिए बनाया गया · प्रोटोटाइप डेमो",
        "language": "Language / भाषा / மொழி",
    },
    "ta": {
        "subtitle": "விவசாயிகளை சந்தை, நோய் கண்டறிதல், வானிலை, கடன் மற்றும் சமூகத்துடன் இணைத்தல்",
        "tab_dash": "📊 டாஷ்போர்டு", "tab_scan": "🔬 நோய் ஸ்கேனர்", "tab_market": "🛒 சந்தை",
        "tab_logistics": "🚚 தளவாடம்", "tab_weather": "🌦️ வானிலை", "tab_advisory": "🌱 பயிர் ஆலோசனை",
        "tab_finance": "💰 நிதி", "tab_community": "💬 சமூகம்",
        "active_listings": "செயலில் உள்ள பட்டியல்கள்", "orders_logistics": "தளவாடத்தில் ஆர்டர்கள்",
        "scans_performed": "செய்யப்பட்ட பயிர் ஸ்கேன்கள்", "community_posts": "சமூக இடுகைகள்",
        "recent_orders": "சமீபத்திய ஆர்டர்கள்", "no_orders": "இன்னும் ஆர்டர்கள் இல்லை",
        "price_trends": "சந்தை விலை போக்குகள் (கிலோவுக்கு)", "buyer": "வாங்குபவர்", "status": "நிலை",
        "scan_title": "பயிர் நோய் ஸ்கேனர்", "upload_leaf": "இலை புகைப்படத்தை பதிவேற்றவும்",
        "running_inference": "பகுப்பாய்வு நடைபெறுகிறது...", "top3": "முதல்-3 மாடல் கணிப்புகளை காண்க",
        "confidence": "நம்பிக்கை நிலை",
        "model_note": "மாடல்: PlantVillage தரவுத்தொகுப்பில் பயிற்சி பெற்ற MobileNetV2 (38 வகைகள், ~95% துல்லியம்). நிபுணர் நோயறிதலுக்கு மாற்றாக இல்லை.",
        "scan_history": "ஸ்கேன் வரலாறு", "no_scans": "இன்னும் ஸ்கேன்கள் இல்லை — மேலே ஒரு புகைப்படத்தை பதிவேற்றவும்",
        "list_produce": "உங்கள் விளைபொருளை பட்டியலிடவும்", "crop": "பயிர்", "quantity": "அளவு", "price": "விலை",
        "location": "இடம்", "add_listing": "+ பட்டியல் சேர்க்க",
        "fill_error": "தயவுசெய்து பயிர், அளவு மற்றும் விலையை நிரப்பவும்", "listed_success": "{crop} பட்டியலிடப்பட்டது",
        "available_produce": "கிடைக்கும் விளைபொருள்", "search_crop": "🔍 பயிர் மூலம் தேடு",
        "filter_loc": "📍 இடம் மூலம் வடிகட்டு", "export_csv": "⬇️ CSV ஏற்றுமதி",
        "buy_contact": "வாங்க / தொடர்பு கொள்ள", "order_success": "ஆர்டர் வைக்கப்பட்டது — தளவாடம் தாவலைப் பார்க்கவும்",
        "no_match": "உங்கள் தேடல்/வடிகட்டலுடன் பொருந்தும் பட்டியல்கள் இல்லை",
        "active_orders": "செயலில் உள்ள ஆர்டர்கள்", "no_orders_market": "இன்னும் ஆர்டர்கள் இல்லை — சந்தையில் ஏதாவது வாங்கவும்",
        "est_delivery": "மதிப்பிடப்பட்ட டெலிவரி", "advance_status": "நிலையை மேம்படுத்து", "complete": "முடிந்தது",
        "weather_title": "வானிலை முன்னறிவிப்பு", "enter_location": "உங்கள் இடத்தை உள்ளிடவும்",
        "get_forecast": "முன்னறிவிப்பைப் பெறவும்",
        "weather_fail": "நேரடி வானிலையை பெற முடியவில்லை (இட எழுத்துப்பிழை அல்லது இணைய இணைப்பை சரிபார்க்கவும்).",
        "temperature": "வெப்பநிலை", "humidity": "ஈரப்பதம்", "wind": "காற்று", "precipitation": "மழைப்பொழிவு",
        "outlook_5day": "5-நாள் முன்னறிவிப்பு",
        "advisory_title": "பயிர் பரிந்துரை", "season": "பருவம்", "soil": "மண் வகை",
        "get_recommendations": "பரிந்துரைகளைப் பெறவும்", "recommended_for": "**{season}** பருவத்திற்கு **{soil}** மண்ணில் பரிந்துரைக்கப்படும் பயிர்கள்:",
        "no_advisory_match": "குறிப்பிட்ட பொருத்தம் இல்லை — உள்ளூர் வேளாண் அலுவலகத்தை அணுகவும்.",
        "finance_title": "விவசாய கடன் / EMI கால்குலேட்டர்", "loan_amount": "கடன் தொகை (₹)",
        "interest_rate": "ஆண்டு வட்டி விகிதம் (%)", "tenure": "காலம் (மாதங்கள்)",
        "est_emi": "மதிப்பிடப்பட்ட மாத EMI",
        "repayment_caption": "மொத்த திருப்பிச் செலுத்துதல்: ₹{total:,.0f} · மொத்த வட்டி: ₹{interest:,.0f}",
        "finance_disclaimer": "இது ஒரு தோராயமான மதிப்பீடு மட்டுமே — உண்மையான விகிதங்கள் கடன் வழங்குபவர் மற்றும் அரசு விவசாய கடன் திட்டங்களைப் பொறுத்தது (எ.கா. கிசான் கிரெடிட் கார்டு).",
        "community_title": "விவசாயி சமூக பலகை", "your_name": "உங்கள் பெயர்",
        "share_placeholder": "ஒரு கேள்வி, குறிப்பு அல்லது புதுப்பிப்பை பகிரவும்", "post_btn": "இடுகையிடு", "posted": "இடுகையிடப்பட்டது!",
        "footer": "அக்ரி பிரிட்ஜ் ஹேக்கத்தானுக்காக உருவாக்கப்பட்டது · முன்மாதிரி டெமோ",
        "language": "Language / भाषा / மொழி",
    },
}

if "lang" not in st.session_state:
    st.session_state.lang = "en"


def t(key, **kwargs):
    s = T[st.session_state.lang].get(key, T["en"].get(key, key))
    return s.format(**kwargs) if kwargs else s


STEPS_TR = {
    "en": ["Order Placed", "Farmer Confirmed", "In Transit", "Delivered"],
    "hi": ["ऑर्डर दिया गया", "किसान द्वारा पुष्टि", "परिवहन में", "डिलीवर हो गया"],
    "ta": ["ஆர்டர் வைக்கப்பட்டது", "விவசாயி உறுதிப்படுத்தினார்", "போக்குவரத்தில்", "டெலிவரி செய்யப்பட்டது"],
}

SEASONS = ["Kharif (Monsoon)", "Rabi (Winter)", "Zaid (Summer)"]
SOILS = ["Clay", "Loamy", "Sandy"]
SEASON_TR = {
    "en": {s: s for s in SEASONS},
    "hi": dict(zip(SEASONS, ["खरीफ (मानसून)", "रबी (सर्दी)", "जायद (गर्मी)"])),
    "ta": dict(zip(SEASONS, ["காரீஃப் (பருவமழை)", "ரபி (குளிர்காலம்)", "ஜாயத் (கோடை)"])),
}
SOIL_TR = {
    "en": {s: s for s in SOILS},
    "hi": dict(zip(SOILS, ["चिकनी मिट्टी", "दोमट मिट्टी", "बलुई मिट्टी"])),
    "ta": dict(zip(SOILS, ["களிமண்", "வண்டல் மண்", "மணல் மண்"])),
}
CROP_TR = {
    "Rice": {"hi": "चावल", "ta": "அரிசி"}, "Jute": {"hi": "जूट", "ta": "சணல்"},
    "Sugarcane": {"hi": "गन्ना", "ta": "கரும்பு"}, "Cotton": {"hi": "कपास", "ta": "பருத்தி"},
    "Maize": {"hi": "मक्का", "ta": "சோளம்"}, "Soybean": {"hi": "सोयाबीन", "ta": "சோயாபீன்"},
    "Groundnut": {"hi": "मूंगफली", "ta": "நிலக்கடலை"},
    "Bajra (Pearl Millet)": {"hi": "बाजरा", "ta": "கம்பு"},
    "Watermelon": {"hi": "तरबूज", "ta": "தர்பூசணி"}, "Wheat": {"hi": "गेहूं", "ta": "கோதுமை"},
    "Gram (Chickpea)": {"hi": "चना", "ta": "கொண்டைக்கடலை"}, "Mustard": {"hi": "सरसों", "ta": "கடுகு"},
    "Barley": {"hi": "जौ", "ta": "பார்லி"}, "Peas": {"hi": "मटर", "ta": "பட்டாணி"},
    "Vegetables (Okra, Gourd)": {"hi": "सब्जियां (भिंडी, लौकी)", "ta": "காய்கறிகள் (வெண்டைக்காய், சுரைக்காய்)"},
    "Muskmelon": {"hi": "खरबूजा", "ta": "முலாம்பழம்"}, "Cucumber": {"hi": "खीरा", "ta": "வெள்ளரிக்காய்"},
    "Fodder crops": {"hi": "चारा फसलें", "ta": "தீவன பயிர்கள்"},
}


def tr_crop(name):
    lang = st.session_state.lang
    if lang == "en":
        return name
    return CROP_TR.get(name, {}).get(lang, name)


# ---------------------------------------------------------------------------
# Disease model config
# ---------------------------------------------------------------------------
MODEL_NAME = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
KEYWORD_TIPS = {
    "en": [
        ("healthy", "ok", "No action needed — maintain regular watering and monitor weekly."),
        ("blight", "bad", "Remove and destroy infected leaves, apply a copper-based fungicide, improve field drainage and spacing for airflow."),
        ("rust", "warn", "Apply a triazole fungicide, remove volunteer/nearby host plants, and favor rust-resistant varieties next season."),
        ("mildew", "warn", "Increase airflow between plants, apply a sulfur or potassium-bicarbonate spray, avoid overhead watering."),
        ("scab", "warn", "Prune for airflow, rake and destroy fallen leaves, apply fungicide at bud break next season."),
        ("rot", "bad", "Remove and destroy affected tissue, avoid overhead irrigation, improve drainage, apply appropriate fungicide."),
        ("spot", "warn", "Remove affected leaves, avoid overhead watering, apply copper-based bactericide/fungicide, rotate crops."),
        ("mite", "warn", "Introduce predatory mites or apply insecticidal soap / miticide; avoid drought stress which favors mites."),
        ("mold", "warn", "Improve ventilation and reduce humidity around plants, remove affected foliage, apply fungicide if severe."),
        ("mosaic", "bad", "No cure — remove and destroy infected plants, control aphid vectors, use certified virus-free seed/planting material."),
        ("curl", "bad", "Control whitefly vectors (the usual carrier), remove infected plants, use resistant varieties and reflective mulch."),
        ("greening", "bad", "No cure — remove infected trees, control psyllid insect vectors aggressively, use certified disease-free nursery stock."),
        ("bacterial", "bad", "Use disease-free seed, apply copper-based bactericide, avoid working in fields when leaves are wet."),
    ],
    "hi": [
        ("healthy", "ok", "किसी कार्रवाई की आवश्यकता नहीं — नियमित सिंचाई करें और साप्ताहिक निगरानी रखें।"),
        ("blight", "bad", "संक्रमित पत्तियों को हटाकर नष्ट करें, कॉपर-आधारित फफूंदनाशक लगाएं, खेत की जल निकासी और हवा का प्रवाह सुधारें।"),
        ("rust", "warn", "ट्राईज़ोल फफूंदनाशक लगाएं, आस-पास के मेज़बान पौधों को हटाएं, अगले सीज़न में रोग-प्रतिरोधी किस्में चुनें।"),
        ("mildew", "warn", "पौधों के बीच हवा का प्रवाह बढ़ाएं, सल्फर स्प्रे लगाएं, ऊपर से पानी देने से बचें।"),
        ("scab", "warn", "हवा के प्रवाह के लिए छंटाई करें, गिरी हुई पत्तियों को नष्ट करें, अगले सीज़न में फफूंदनाशक लगाएं।"),
        ("rot", "bad", "प्रभावित हिस्सों को हटाकर नष्ट करें, ऊपर से सिंचाई से बचें, जल निकासी सुधारें, उपयुक्त फफूंदनाशक लगाएं।"),
        ("spot", "warn", "प्रभावित पत्तियों को हटाएं, ऊपर से पानी देने से बचें, कॉपर-आधारित दवा लगाएं, फसल चक्र अपनाएं।"),
        ("mite", "warn", "शिकारी माइट्स डालें या कीटनाशक साबुन/माइटिसाइड लगाएं; सूखे तनाव से बचें।"),
        ("mold", "warn", "हवादार बनाएं और नमी कम करें, प्रभावित पत्तियों को हटाएं, गंभीर होने पर फफूंदनाशक लगाएं।"),
        ("mosaic", "bad", "कोई इलाज नहीं — संक्रमित पौधों को हटाकर नष्ट करें, एफिड वाहकों को नियंत्रित करें, प्रमाणित वायरस-मुक्त बीज का उपयोग करें।"),
        ("curl", "bad", "व्हाइटफ्लाई वाहकों को नियंत्रित करें, संक्रमित पौधों को हटाएं, प्रतिरोधी किस्मों का उपयोग करें।"),
        ("greening", "bad", "कोई इलाज नहीं — संक्रमित पेड़ों को हटाएं, कीट वाहकों को सख्ती से नियंत्रित करें।"),
        ("bacterial", "bad", "रोग-मुक्त बीज का उपयोग करें, कॉपर-आधारित जीवाणुनाशक लगाएं, पत्तियां गीली होने पर खेत में काम करने से बचें।"),
    ],
    "ta": [
        ("healthy", "ok", "எந்த நடவடிக்கையும் தேவையில்லை — வழக்கமான நீர்ப்பாசனம் செய்து வாரந்தோறும் கண்காணிக்கவும்."),
        ("blight", "bad", "பாதிக்கப்பட்ட இலைகளை அகற்றி அழிக்கவும், காப்பர் அடிப்படையிலான பூஞ்சைக்கொல்லி பயன்படுத்தவும், வடிகால் மற்றும் காற்றோட்டத்தை மேம்படுத்தவும்."),
        ("rust", "warn", "ட்ரையசோல் பூஞ்சைக்கொல்லியை பயன்படுத்தவும், அருகிலுள்ள ஆதார தாவரங்களை அகற்றவும், எதிர்ப்பு வகைகளை தேர்வு செய்யவும்."),
        ("mildew", "warn", "தாவரங்களுக்கு இடையே காற்றோட்டத்தை அதிகரிக்கவும், சல்பர் தெளிப்பான் பயன்படுத்தவும், மேலிருந்து நீர் பாய்ச்சுவதை தவிர்க்கவும்."),
        ("scab", "warn", "காற்றோட்டத்திற்காக கத்தரிக்கவும், விழுந்த இலைகளை அகற்றி அழிக்கவும், அடுத்த பருவத்தில் பூஞ்சைக்கொல்லி பயன்படுத்தவும்."),
        ("rot", "bad", "பாதிக்கப்பட்ட பகுதிகளை அகற்றி அழிக்கவும், மேலிருந்து நீர்ப்பாசனத்தை தவிர்க்கவும், வடிகாலை மேம்படுத்தவும்."),
        ("spot", "warn", "பாதிக்கப்பட்ட இலைகளை அகற்றவும், மேலிருந்து நீர் பாய்ச்சுவதை தவிர்க்கவும், காப்பர் மருந்தை பயன்படுத்தவும்."),
        ("mite", "warn", "வேட்டையாடும் நுளம்புகளை அறிமுகப்படுத்தவும் அல்லது பூச்சிக்கொல்லி சோப்பை பயன்படுத்தவும்; வறட்சி அழுத்தத்தை தவிர்க்கவும்."),
        ("mold", "warn", "காற்றோட்டத்தை மேம்படுத்தி ஈரப்பதத்தை குறைக்கவும், பாதிக்கப்பட்ட இலைகளை அகற்றவும்."),
        ("mosaic", "bad", "மருந்து இல்லை — பாதிக்கப்பட்ட தாவரங்களை அகற்றி அழிக்கவும், பேன் பூச்சிகளை கட்டுப்படுத்தவும்."),
        ("curl", "bad", "வெள்ளை ஈக்களை கட்டுப்படுத்தவும், பாதிக்கப்பட்ட தாவரங்களை அகற்றவும், எதிர்ப்பு வகைகளை பயன்படுத்தவும்."),
        ("greening", "bad", "மருந்து இல்லை — பாதிக்கப்பட்ட மரங்களை அகற்றவும், பூச்சி வாககங்களை கடுமையாக கட்டுப்படுத்தவும்."),
        ("bacterial", "bad", "நோய் இல்லா விதைகளை பயன்படுத்தவும், காப்பர் மருந்தை பயன்படுத்தவும், இலைகள் ஈரமாக இருக்கும்போது வேலை தவிர்க்கவும்."),
    ],
}
GENERIC_TIP = {
    "en": ("warn", "Isolate the affected plant, monitor closely, and consult a local agricultural extension office for confirmation."),
    "hi": ("warn", "प्रभावित पौधे को अलग करें, बारीकी से निगरानी करें, और पुष्टि के लिए स्थानीय कृषि विस्तार कार्यालय से सलाह लें।"),
    "ta": ("warn", "பாதிக்கப்பட்ட தாவரத்தை தனிமைப்படுத்தவும், உன்னிப்பாக கண்காணிக்கவும், உள்ளூர் வேளாண் அலுவலகத்தை அணுகவும்."),
}


@st.cache_resource(show_spinner="Loading plant disease model (first run only)...")
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return processor, model


def tip_for_label(raw_label: str):
    lang = st.session_state.lang
    label = raw_label.lower()
    for kw, sev, tip in KEYWORD_TIPS[lang]:
        if kw in label:
            return sev, tip
    return GENERIC_TIP[lang]


def pretty_label(raw_label: str) -> str:
    parts = raw_label.replace("___", "|").replace("_", " ").split("|")
    if len(parts) == 2:
        crop, disease = parts
        return f"{crop.strip()} — {disease.strip()}"
    return raw_label.replace("_", " ")


SEVERITY_COLOR = {"ok": "green", "warn": "orange", "bad": "red"}

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
def analyze_leaf(image: Image.Image, top_k: int = 3):
    processor, model = load_model()
    img = image.convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]

    top_probs, top_idx = torch.topk(probs, k=min(top_k, probs.shape[0]))
    id2label = model.config.id2label
    results = [
        {"label": id2label[i.item()], "score": float(p)}
        for p, i in zip(top_probs, top_idx)
    ]

    best = results[0]
    severity, tip = tip_for_label(best["label"])
    return {
        "name": pretty_label(best["label"]),
        "severity": severity,
        "tip": tip,
        "confidence": round(best["score"] * 100, 1),
        "top_k": [{"name": pretty_label(r["label"]), "score": round(r["score"] * 100, 1)} for r in results],
    }


@st.cache_data(ttl=1800, show_spinner=False)
def get_weather(location: str):
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
# Header + language selector
# ---------------------------------------------------------------------------
hc1, hc2 = st.columns([4, 1])
with hc1:
    st.markdown(
        f"<div class='agri-banner'><h1>🌾 AgriBridge</h1><p>{t('subtitle')}</p></div>",
        unsafe_allow_html=True,
    )
with hc2:
    chosen = st.selectbox(t("language"), list(LANGS.keys()),
                           index=list(LANGS.values()).index(st.session_state.lang))
    st.session_state.lang = LANGS[chosen]

tabs = st.tabs([
    t("tab_dash"), t("tab_scan"), t("tab_market"), t("tab_logistics"),
    t("tab_weather"), t("tab_advisory"), t("tab_finance"), t("tab_community"),
])
(tab_dash, tab_scan, tab_market, tab_logistics,
 tab_weather, tab_advisory, tab_finance, tab_community) = tabs

STEPS = STEPS_TR[st.session_state.lang]

# --- Dashboard ---
with tab_dash:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("active_listings"), len(st.session_state.listings))
    c2.metric(t("orders_logistics"), len(st.session_state.orders))
    c3.metric(t("scans_performed"), len(st.session_state.scans))
    c4.metric(t("community_posts"), len(st.session_state.posts))

    st.subheader(t("recent_orders"))
    if st.session_state.orders:
        for o in st.session_state.orders:
            st.write(f"**{o['item']}** — {t('buyer')}: {o['buyer']} — {t('status')}: *{STEPS[o['status']]}*")
    else:
        st.caption(t("no_orders"))

    st.subheader(t("price_trends"))
    price_df = pd.DataFrame({
        "Day": pd.date_range(end=datetime.now(), periods=7).strftime("%a"),
        "Tomatoes": [24, 25, 26, 27, 26, 28, 28],
        "Rice": [64, 64, 65, 65, 66, 65, 65],
        "Onions": [25, 24, 23, 23, 22, 22, 22],
    }).set_index("Day")
    st.line_chart(price_df)

# --- Disease Scanner ---
with tab_scan:
    st.subheader(t("scan_title"))
    uploaded = st.file_uploader(t("upload_leaf"), type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        image = Image.open(uploaded)
        col1, col2 = st.columns([1, 1.4])
        with col1:
            st.image(image, use_container_width=True)
        with col2:
            with st.spinner(t("running_inference")):
                result = analyze_leaf(image)
            st.session_state.scans.append(result)
            st.markdown(f":{SEVERITY_COLOR[result['severity']]}[**{result['name']}**]")
            st.write(result["tip"])
            st.caption(f"{t('confidence')}: {result['confidence']}%")
            with st.expander(t("top3")):
                for r in result["top_k"]:
                    st.write(f"{r['name']} — {r['score']}%")
    st.caption(t("model_note"))

    st.subheader(t("scan_history"))
    if st.session_state.scans:
        for s in reversed(st.session_state.scans):
            with st.container(border=True):
                st.markdown(f":{SEVERITY_COLOR[s['severity']]}[**{s['name']}**]  ·  {s['confidence']}%")
                st.caption(s["tip"])
    else:
        st.caption(t("no_scans"))

# --- Marketplace ---
with tab_market:
    st.subheader(t("list_produce"))
    with st.form("add_listing", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns(4)
        crop = c1.text_input(t("crop"), placeholder="e.g. Wheat")
        qty = c2.text_input(t("quantity"), placeholder="e.g. 100 kg")
        price = c3.text_input(t("price"), placeholder="e.g. ₹30/kg")
        loc = c4.text_input(t("location"), placeholder="e.g. Salem, TN")
        if st.form_submit_button(t("add_listing"), type="primary"):
            if crop and qty and price:
                st.session_state.listings.insert(
                    0, {"crop": crop, "qty": qty, "price": price, "farmer": "You", "loc": loc or "—"}
                )
                st.success(t("listed_success", crop=crop))
            else:
                st.error(t("fill_error"))

    st.subheader(t("available_produce"))
    fc1, fc2, fc3 = st.columns([2, 2, 1])
    search = fc1.text_input(t("search_crop"), placeholder="e.g. rice")
    loc_filter = fc2.text_input(t("filter_loc"), placeholder="e.g. TN")
    filtered = [
        l for l in st.session_state.listings
        if search.lower() in l["crop"].lower()
        and loc_filter.lower() in l["loc"].lower()
    ]
    csv = pd.DataFrame(st.session_state.listings).to_csv(index=False).encode("utf-8")
    fc3.download_button(t("export_csv"), csv, "agribridge_listings.csv", "text/csv")

    cols = st.columns(3)
    for i, l in enumerate(filtered):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"**{l['crop']}**")
                st.caption(f"{l['qty']} · {l['price']}")
                st.caption(f"{l['farmer']} · {l['loc']}")
                if st.button(t("buy_contact"), key=f"buy_{i}_{l['crop']}"):
                    st.session_state.orders.insert(
                        0, {"item": f"{l['crop']} ({l['qty']})", "buyer": "You", "status": 0, "placed": datetime.now()}
                    )
                    st.success(t("order_success"))
    if not filtered:
        st.caption(t("no_match"))

# --- Logistics ---
with tab_logistics:
    st.subheader(t("active_orders"))
    if not st.session_state.orders:
        st.caption(t("no_orders_market"))
    for idx, o in enumerate(st.session_state.orders):
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"**{o['item']}**")
            eta = o["placed"] + timedelta(days=4 - o["status"])
            c1.caption(f"{t('buyer')}: {o['buyer']} · {t('est_delivery')}: {eta.strftime('%d %b')}")
            if o["status"] < 3:
                if c2.button(t("advance_status"), key=f"adv_{idx}"):
                    o["status"] += 1
            else:
                c2.success(t("complete"))
            st.progress(o["status"] / (len(STEPS) - 1), text=STEPS[o["status"]])

# --- Weather ---
with tab_weather:
    st.subheader(t("weather_title"))
    loc = st.text_input(t("enter_location"), value="Chennai, India")
    if st.button(t("get_forecast")) or loc:
        data = get_weather(loc)
        if data is None:
            st.warning(t("weather_fail"))
        else:
            cur = data["current"]
            st.markdown(f"**{data['place']}**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(t("temperature"), f"{cur.get('temperature_2m', '—')}°C")
            c2.metric(t("humidity"), f"{cur.get('relative_humidity_2m', '—')}%")
            c3.metric(t("wind"), f"{cur.get('wind_speed_10m', '—')} km/h")
            c4.metric(t("precipitation"), f"{cur.get('precipitation', '—')} mm")

            daily = data["daily"]
            if daily:
                df = pd.DataFrame({
                    "Date": daily.get("time", []),
                    "Max °C": daily.get("temperature_2m_max", []),
                    "Min °C": daily.get("temperature_2m_min", []),
                    "Rain chance %": daily.get("precipitation_probability_max", []),
                }).set_index("Date")
                st.write(t("outlook_5day"))
                st.dataframe(df, use_container_width=True)
                st.line_chart(df[["Max °C", "Min °C"]])

# --- Crop Advisory ---
with tab_advisory:
    st.subheader(t("advisory_title"))
    lang = st.session_state.lang
    c1, c2 = st.columns(2)
    season = c1.selectbox(t("season"), SEASONS, format_func=lambda s: SEASON_TR[lang][s])
    soil = c2.selectbox(t("soil"), SOILS, format_func=lambda s: SOIL_TR[lang][s])
    if st.button(t("get_recommendations"), type="primary"):
        match = next((c for s, so, c in CROP_RULES if s == season and so == soil), [])
        if match:
            st.success(t("recommended_for", season=SEASON_TR[lang][season], soil=SOIL_TR[lang][soil]))
            for crop in match:
                st.write(f"🌾 {tr_crop(crop)}")
        else:
            st.info(t("no_advisory_match"))

# --- Finance ---
with tab_finance:
    st.subheader(t("finance_title"))
    c1, c2, c3 = st.columns(3)
    principal = c1.number_input(t("loan_amount"), min_value=1000, value=50000, step=1000)
    rate = c2.number_input(t("interest_rate"), min_value=1.0, value=7.0, step=0.5)
    months = c3.number_input(t("tenure"), min_value=1, value=12, step=1)
    emi = emi_calculator(principal, rate, months)
    st.metric(t("est_emi"), f"₹{emi:,.0f}")
    st.caption(t("repayment_caption", total=emi * months, interest=emi * months - principal))
    st.caption(t("finance_disclaimer"))

# --- Community ---
with tab_community:
    st.subheader(t("community_title"))
    with st.form("new_post", clear_on_submit=True):
        name = st.text_input(t("your_name"), placeholder="e.g. A. Farmer")
        text = st.text_area(t("share_placeholder"))
        if st.form_submit_button(t("post_btn")):
            if text.strip():
                st.session_state.posts.insert(0, {"author": name or "Anonymous", "text": text, "time": datetime.now()})
                st.success(t("posted"))

    for p in st.session_state.posts:
        with st.container(border=True):
            st.markdown(f"**{p['author']}** · {p['time'].strftime('%d %b, %I:%M %p')}")
            st.write(p["text"])

st.markdown("---")
st.caption(t("footer"))
