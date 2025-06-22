import streamlit as st
import tempfile
import speech_recognition as sr
from difflib import SequenceMatcher
import re
import os
from gtts import gTTS
import io
from PIL import Image
import time
import numpy as np
from collections import defaultdict
import random
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
from typing import Union
from scipy.io.wavfile import write


# --- Theme ve Genel Görünüm Ayarları ---
def set_ui_theme():
    st.set_page_config(
        page_title="PoetryMaster Pro - Şiir Ezberleme Uygulaması",
        page_icon="📖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Özel CSS
    st.markdown("""
    <style>
        :root {
            --primary-color: #4a6fa5;
            --secondary-color: #166088;
            --accent-color: #4cb963;
            --error-color: #ff3a3a;
        }
        .main {
            background-color: #f8f9fa;
        }
        .sidebar .sidebar-content {
            background-color: #ffffff;
            box-shadow: 2px 0 10px rgba(0,0,0,0.1);
        }
        .stButton>button {
            border-radius: 20px;
            padding: 10px 24px;
            font-weight: 500;
            transition: all 0.3s;
            border: 1px solid var(--primary-color);
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .primary-button {
            background-color: var(--primary-color) !important;
            color: white !important;
        }
        .secondary-button {
            background-color: white !important;
            color: var(--primary-color) !important;
        }
        .progress-bar {
            height: 10px;
            border-radius: 5px;
        }
        .poem-line {
            font-size: 1.4rem;
            line-height: 2.2rem;
            margin: 1.5rem 0;
            padding: 1.2rem;
            border-radius: 12px;
            background-color: #ffffff;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }
        .success-box {
            background-color: #e6f7ee;
            border-left: 4px solid var(--accent-color);
        }
        .error-box {
            background-color: #ffebee;
            border-left: 4px solid var(--error-color);
        }
        .user-type-card {
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: all 0.3s;
        }
        .user-type-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        .word-tag {
            display: inline-block;
            padding: 0.5rem 1rem;
            margin: 0.25rem;
            border-radius: 20px;
            background-color: #e3f2fd;
            cursor: grab;
            transition: all 0.2s;
        }
        .word-tag:hover {
            background-color: #bbdefb;
            transform: scale(1.05);
        }
        .selected-word {
            background-color: #a5d6a7 !important;
        }
        .word-container {
            min-height: 120px;
            padding: 1rem;
            border-radius: 12px;
            background-color: #f5f5f5;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)


set_ui_theme()

# --- Şiir Verisi ---
siirler = {
    "İstiklal Marşı (ilk 10 dize)": {
        "content": [
            "Korkma, sönmez bu şafaklarda yüzen al sancak;",
            "Sönmeden yurdumun üstünde tüten en son ocak.",
            "O benim milletimin yıldızıdır, parlayacak;",
            "O benimdir, o benim milletimindir ancak.",
            "Çatma, kurban olayım çehreni ey nazlı hilâl!",
            "Kahraman ırkıma bir gül… ne bu şiddet bu celâl?",
            "Sana olmaz dökülen kanlarımız sonra helâl,",
            "Hakkıdır, Hakk'a tapan, milletimin istiklâl.",
            "Ben ezelden beridir hür yaşadım, hür yaşarım.",
            "Hangi çılgın bana zincir vuracakmış? Şaşarım!"
        ],
        "author": "Mehmet Akif Ersoy",
        "bg_color": "#e3f2fd",
        "difficulty": "Orta"
    },
    "Sessiz Gemi": {
        "content": [
            "Artık demir almak günü gelmişse zamandan,",
            "Meçhule giden bir gemi kalkar bu limandan.",
            "Hiç yolcusu yokmuş gibi sessizce alır yol;",
            "Sallanmaz o kalkışta ne mendil ne de bir kol."
        ],
        "author": "Yahya Kemal Beyatlı",
        "bg_color": "#fff8e1",
        "difficulty": "Kolay"
    },
    "Yaş Otuz Beş": {
        "content": [
            "Yaş otuz beş! Yolun yarısı eder.",
            "Dante gibi ortasındayız ömrün.",
            "Delikanlı çağımızdaki cevher,",
            "Yalvarmak, yakarmak nafile bugün,",
            "Gözünün yaşına bakmadan gider."
        ],
        "author": "Cahit Sıtkı Tarancı",
        "bg_color": "#fce4ec",
        "difficulty": "Zor"
    }
}


# --- Yardımcı Fonksiyonlar ---
@st.cache_data(ttl=3600)
def clean_text(text):
    return re.sub(r'[^\w\s]', '', text.lower())


def check_similarity(original, spoken, threshold=0.75):
    orig = clean_text(original)
    user = clean_text(spoken)
    similarity = SequenceMatcher(None, orig, user).ratio()
    return similarity, similarity >= threshold


def record_audio():
    st.warning("Lütfen mikrofon erişimine izin verin")
    audio_data = None

    def audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
        nonlocal audio_data
        audio_data = frame.to_ndarray()
        return frame

    ctx = webrtc_streamer(
        key="poetry-recorder",
        mode=WebRtcMode.SENDONLY,
        audio_frame_callback=audio_frame_callback,
        media_stream_constraints={
            "audio": {
                "sampleRate": 44100,
                "channelCount": 1
            },
            "video": False
        },
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )

    if ctx.state.playing and audio_data is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            write(tmp_file.name, 44100, audio_data)
            return tmp_file.name
    return None


@st.cache_data(ttl=3600)
def text_to_speech(text, lang='tr', slow=False):
    try:
        tts = gTTS(text=text, lang=lang, slow=slow)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.error("Ses oluşturulurken bir hata oluştu.")
        return None


def shuffle_words(line):
    words = re.findall(r"\w+|[^\w\s]", line)
    shuffled = words.copy()
    if len(shuffled) > 3:
        while True:
            random.shuffle(shuffled)
            if " ".join(shuffled) != line:
                break
    return shuffled, words


def generate_poem_background(poem_data):
    img = Image.new('RGB', (800, 200), color=poem_data["bg_color"])
    return img


# --- Oturum Yönetimi ---
def init_session():
    if 'line_index' not in st.session_state:
        st.session_state.line_index = 0
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = "show"
    if 'threshold' not in st.session_state:
        st.session_state.threshold = 0.75
    if 'user_age' not in st.session_state:
        st.session_state.user_age = "Yetişkin (18-65)"
    if 'completed_lines' not in st.session_state:
        st.session_state.completed_lines = []
    if 'selected_poem' not in st.session_state:
        st.session_state.selected_poem = list(siirler.keys())[0]
    if 'selected_words' not in st.session_state:
        st.session_state.selected_words = []
    if 'word_scores' not in st.session_state:
        st.session_state.word_scores = defaultdict(int)
    if 'audio_enabled' not in st.session_state:
        st.session_state.audio_enabled = True


init_session()


# --- Kullanıcı Arayüzü Bileşenleri ---
def user_profile_card():
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h3 style="color: #4a4a4a; margin-bottom: 0.5rem;">👤 Kullanıcı Profili</h3>
            <div style="background-color: #ffffff; border-radius: 12px; padding: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <p style="margin-bottom: 0.5rem;"><strong>Kullanıcı Tipi:</strong> {}</p>
                <p style="margin-bottom: 0.5rem;"><strong>Zorluk Seviyesi:</strong> {}</p>
                <p style="margin-bottom: 0;"><strong>Ses Özellikleri:</strong> {}</p>
            </div>
        </div>
        """.format(
            st.session_state.user_age,
            st.session_state.threshold,
            "Açık" if st.session_state.audio_enabled else "Kapalı"
        ),
            unsafe_allow_html=True)


def poem_selection_card():
    st.session_state.selected_poem = st.sidebar.selectbox(
        "📜 Şiir Seçin",
        list(siirler.keys()),
        key="poem_selector"
    )

    poem_data = siirler[st.session_state.selected_poem]
    st.sidebar.markdown(f"""
    <div style="background-color: {poem_data['bg_color']}; border-radius: 12px; padding: 1rem; margin-top: 1rem;">
        <p style="font-weight: 500; margin-bottom: 0.5rem;">Şair: <strong>{poem_data['author']}</strong></p>
        <p style="margin-bottom: 0.5rem;">Zorluk: <strong>{poem_data.get('difficulty', 'Belirtilmemiş')}</strong></p>
        <p style="margin-bottom: 0;">Toplam Satır: <strong>{len(poem_data['content'])}</strong></p>
    </div>
    """, unsafe_allow_html=True)


def progress_tracker():
    poem_data = siirler[st.session_state.selected_poem]
    progress = (st.session_state.line_index) / len(poem_data["content"])

    st.markdown(f"""
    <div style="margin: 1.5rem 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span>İlerleme Durumu</span>
            <span>{st.session_state.line_index}/{len(poem_data['content'])} satır</span>
        </div>
        <div class="progress-bar" style="background-color: #e0e0e0;">
            <div style="width: {progress * 100}%; height: 100%; background-color: var(--accent-color); border-radius: 5px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def show_line(poem_data):
    current_line = poem_data["content"][st.session_state.line_index]

    st.markdown(f"""
    <div class="poem-line" style="background-color: {poem_data['bg_color']};">
        {current_line}
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("🔊 Sesli Oku", use_container_width=True, disabled=not st.session_state.audio_enabled):
            audio = text_to_speech(current_line)
            if audio:
                st.audio(audio)
    with col2:
        if st.button("🎤 Sesli Test", type="primary", use_container_width=True):
            st.session_state.current_mode = "test"
            st.rerun()
    with col3:
        if st.button("🧩 Kelime Sırala", type="secondary", use_container_width=True):
            st.session_state.current_mode = "word_sort"
            st.rerun()

    with st.expander("📌 Ezberleme Teknikleri"):
        st.markdown("""
        - **Yüksek sesle okuyun**: Duyarak öğrenmek daha etkilidir
        - **Anlamını düşünün**: Şiirin anlamını anlamak ezberi kolaylaştırır
        - **Tekrarlayın**: 20 dakika sonra ve 1 gün sonra tekrar yapın
        """)

    if (st.session_state.line_index > 0 and
            st.session_state.line_index % 4 == 0 and
            st.session_state.current_mode == "show"):
        st.markdown("---")
        if st.button(f"🧩 {st.session_state.line_index - 3}-{st.session_state.line_index} arası grup testi yap",
                     use_container_width=True):
            st.session_state.current_mode = "group_test"
            st.rerun()


def test_line(poem_data):
    current_line = poem_data["content"][st.session_state.line_index]

    st.markdown(f"""
    <div style="background-color: #f5f5f5; border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem;">
        <h4 style="color: #4a4a4a; margin-top: 0;">🎤 Aşağıdaki satırı ezberden okuyun:</h4>
        <div style="font-size: 1.2rem; color: #666; font-style: italic;">❓❓❓ (Satır gizleniyor)</div>
    </div>
    """, unsafe_allow_html=True)

    wav_file = record_audio()
    if not wav_file:
        st.button("🔁 Tekrar Dene", key="retry_recording")
        return

    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(wav_file) as source:
            audio = recognizer.record(source)
        spoken_text = recognizer.recognize_google(audio, language="tr-TR")

        st.markdown(f"""
        <div style="background-color: #ffffff; border-radius: 12px; padding: 1.5rem; margin: 1rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            <h4 style="margin-top: 0;">🗣️ Söylediğiniz:</h4>
            <p style="font-size: 1.1rem;">{spoken_text}</p>
        </div>
        """, unsafe_allow_html=True)

        similarity, is_correct = check_similarity(current_line, spoken_text, st.session_state.threshold)

        if is_correct:
            st.session_state.word_scores[st.session_state.line_index] = similarity
            st.markdown(f"""
            <div class="success-box" style="padding: 1.5rem; border-radius: 12px; margin: 1rem 0;">
                <h4 style="color: var(--accent-color); margin-top: 0;">✅ Tebrikler! Doğru okudunuz!</h4>
                <p>Benzerlik Oranı: <strong>{similarity:.0%}</strong></p>
                <p>Sonraki satıra geçiliyor...</p>
            </div>
            """, unsafe_allow_html=True)

            st.session_state.completed_lines.append(st.session_state.line_index)
            st.session_state.line_index += 1
            st.session_state.current_mode = "show"
            time.sleep(2)
            st.rerun()
        else:
            st.markdown(f"""
            <div class="error-box" style="padding: 1.5rem; border-radius: 12px; margin: 1rem 0;">
                <h4 style="color: var(--error-color); margin-top: 0;">❌ Tekrar denemeniz gerekiyor</h4>
                <p>Benzerlik Oranı: <strong>{similarity:.0%}</strong></p>
                <p>Doğru satır: <em>{current_line}</em></p>
            </div>
            """, unsafe_allow_html=True)

            st.button("🔁 Tekrar Dene", key="retry_test", type="primary")
    except sr.UnknownValueError:
        st.error("Ses anlaşılamadı. Lütfen daha net ve yakından konuşarak tekrar deneyin.")
        st.button("🔁 Tekrar Dene", key="retry_unknown")
    except Exception as e:
        st.error(f"Bir hata oluştu: {str(e)}")
    finally:
        if os.path.exists(wav_file):
            os.remove(wav_file)


def word_sort_test(poem_data):
    current_line = poem_data["content"][st.session_state.line_index]

    if 'shuffled_words' not in st.session_state or st.session_state.get('current_test_line') != current_line:
        st.session_state.shuffled_words, st.session_state.correct_words = shuffle_words(current_line)
        st.session_state.selected_words = []
        st.session_state.current_test_line = current_line

    st.markdown(f"""
    <div style="background-color: #f5f5f5; border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem;">
        <h4 style="color: #4a4a4a; margin-top: 0;">🧩 Kelimeleri Doğru Sıraya Diz</h4>
        <div style="font-size: 1.2rem; color: #666; font-style: italic;">❓❓❓ (Satır gizleniyor)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Kelimeleri doğru sıraya dizin (tıklayarak):")

    if st.session_state.selected_words:
        st.markdown("**Seçtiğiniz sıra:**")
        selected_text = " ".join([
                                     f'<span style="display: inline-block; padding: 0.3rem 0.6rem; margin: 0.2rem; background-color: #a5d6a7; border-radius: 12px;">{word}</span>'
                                     for word in st.session_state.selected_words])
        st.markdown(selected_text, unsafe_allow_html=True)

    st.markdown("**Kullanılabilir kelimeler:**")
    available_words = [w for w in st.session_state.shuffled_words if w not in st.session_state.selected_words]

    cols = st.columns(5)
    for i, word in enumerate(available_words):
        col = cols[i % 5]
        if col.button(word, key=f"available_{word}_{i}"):
            st.session_state.selected_words.append(word)
            st.rerun()

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔄 Sıfırla", key="reset_word_sort"):
            st.session_state.selected_words = []
            st.rerun()
    with col2:
        if st.button("✅ Kontrol Et", key="check_word_sort", type="primary"):
            user_answer = " ".join(st.session_state.selected_words)
            correct_answer = " ".join(st.session_state.correct_words)

            if user_answer == correct_answer:
                score = min(1.0,
                            0.7 + (0.3 * (len(st.session_state.selected_words) / len(st.session_state.correct_words))))
                st.session_state.word_scores[st.session_state.line_index] = score

                st.success(f"✅ Doğru! Harika sıraladınız! Puan: {score:.0%}")
                st.session_state.completed_lines.append(st.session_state.line_index)
                st.session_state.line_index += 1
                st.session_state.current_mode = "show"
                time.sleep(2)
                st.rerun()
            else:
                st.error(f"❌ Yanlış sıra. Doğrusu: {correct_answer}")
                st.session_state.word_scores[st.session_state.line_index] = 0.0


def show_stats():
    if st.session_state.word_scores:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Performans İstatistikleri")

        avg_score = np.mean(list(st.session_state.word_scores.values())) if st.session_state.word_scores else 0
        st.sidebar.metric("Ortalama Doğruluk", f"{avg_score:.0%}")

        best_line = max(st.session_state.word_scores.items(), key=lambda x: x[1], default=(None, 0))
        if best_line[0] is not None:
            poem_data = siirler[st.session_state.selected_poem]
            st.sidebar.metric("En İyi Satır",
                              f"{best_line[1]:.0%}",
                              help=f"Satır {best_line[0] + 1}: {poem_data['content'][best_line[0]]}")


# --- Ana Uygulama ---
def main():
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #4a4a4a; margin-bottom: 0.5rem;">📖 PoetryMaster Pro</h1>
        <p style="color: #666; font-size: 1.1rem;">Türkçe şiirleri kolayca ezberleyebileceğiniz akıllı uygulama</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <h3 style="color: #4a4a4a;">⚙️ Ayarlar</h3>
        </div>
        """, unsafe_allow_html=True)

        st.session_state.user_age = st.selectbox(
            "👤 Kullanıcı Tipi",
            ["Çocuk (7-12)", "Genç (13-18)", "Yetişkin (18-65)", "Yaşlı (65+)"],
            key="age_selector"
        )

        st.session_state.threshold = st.slider(
            "🎚️ Zorluk Seviyesi",
            0.5, 1.0, st.session_state.threshold,
            key="difficulty_slider",
            help="Ne kadar yüksek olursa, o kadar hatasız okumanız gerekir"
        )

        st.session_state.audio_enabled = st.checkbox(
            "🔊 Ses özelliklerini aç",
            value=st.session_state.audio_enabled,
            help="Sesli okuma ve ses tanıma özelliklerini etkinleştirir"
        )

        st.markdown("---")
        user_profile_card()
        poem_selection_card()
        show_stats()

        if st.button("🔄 Oturumu Sıfırla", use_container_width=True):
            st.session_state.clear()
            init_session()
            st.rerun()

    poem_data = siirler[st.session_state.selected_poem]

    st.markdown(f"""
    <div style="background-color: {poem_data['bg_color']}; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h2 style="margin-top: 0; color: #333;">{st.session_state.selected_poem}</h2>
                <p style="margin-bottom: 0; font-size: 1.1rem; color: #555;">Şair: {poem_data['author']}</p>
            </div>
            <div style="font-size: 1.2rem; background-color: rgba(255,255,255,0.7); padding: 0.5rem 1rem; border-radius: 20px;">
                {st.session_state.line_index + 1}/{len(poem_data['content'])}. Satır
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    progress_tracker()

    if st.session_state.line_index >= len(poem_data["content"]):
        st.balloons()
        st.success("🎉 Tebrikler! Bu şiiri başarıyla tamamladınız!")
        if st.button("Başka bir şiir seç"):
            st.session_state.line_index = 0
            st.session_state.current_mode = "show"
            st.rerun()
        return

    if st.session_state.current_mode == "show":
        show_line(poem_data)
    elif st.session_state.current_mode == "test":
        test_line(poem_data)
    elif st.session_state.current_mode == "word_sort":
        word_sort_test(poem_data)
    elif st.session_state.current_mode == "group_test":
        st.warning("Grup testi özelliği yakında eklenecek!")
        st.session_state.current_mode = "show"
        st.rerun()


if __name__ == "__main__":
    main()