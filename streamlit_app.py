import streamlit as st
import pandas as pd
import io
import time
import os

st.set_page_config(page_title="Live IDS Monitoring", layout="wide")

# --- Config ---
LOG_FILE = "monitoring_results/ids_detection_log.csv"
POLLING_INTERVAL = 3  # seconds

# --- Session State Setup ---
if "last_line" not in st.session_state:
    st.session_state.last_line = 0
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame()
if "paused" not in st.session_state:
    st.session_state.paused = False
if "initialized" not in st.session_state:
    st.session_state.initialized = False

# --- Sidebar Controls ---
st.sidebar.title("Controls")
st.sidebar.toggle("Pause Updates", key="paused")

# --- Title ---
st.title("📡 Real-Time Intrusion Detection Log")

# --- Function to Read New Rows from CSV ---
def read_new_rows(path, skip_rows):
    try:
        with open(path, "r") as f:
            lines = f.readlines()
            total_lines = len(lines)

            if total_lines > skip_rows:
                new_lines = lines[skip_rows:]
                
                # Drop partial last line (likely being written)
                if not new_lines[-1].endswith('\n'):
                    new_lines = new_lines[:-1]

                # Check if there are valid lines to process
                if new_lines:
                    # Remove duplicates of the header if any (only keep first)
                    header = lines[0]
                    cleaned = [line for line in new_lines if line.strip() != header.strip()]
                    csv_data = header + ''.join(cleaned)

                    df = pd.read_csv(io.StringIO(csv_data))
                    return df, skip_rows + len(new_lines)

    except Exception as e:
        st.error(f"Error reading log: {e}")
    
    return pd.DataFrame(), skip_rows

# --- Live Data Update ---
placeholder = st.empty()

while True:
    if not st.session_state.paused and os.path.exists(LOG_FILE):
        new_df, new_last_line = read_new_rows(LOG_FILE, st.session_state.last_line)

        if not new_df.empty:
            st.session_state.data = pd.concat([st.session_state.data, new_df], ignore_index=True)
            st.session_state.last_line = new_last_line

            # --- Show toast only after initialization ---
            if st.session_state.initialized:
                if 'prediction' in new_df.columns:
                    if not new_df[new_df['prediction'] == 'ATTACK'].empty:
                        st.toast("⚠️ New attack detected!", icon="🚨")
            else:
                st.session_state.initialized = True  # Mark as initialized after first batch

    # --- Display table with highlighted attack rows ---
    with placeholder.container():
        if st.session_state.data.empty:
            st.info("Waiting for data...")
        else:
            st.subheader("📋 Detection Log (Live)")

            # Highlight rows where 'prediction' is 'ATTACK'
            def highlight_attack(row):
                return ['background-color: #ffcccc' if row['prediction'] == 'ATTACK' else '' for _ in row]

            recent_data = st.session_state.data.tail(50)
            styled_df = recent_data.style.apply(highlight_attack, axis=1)
            st.dataframe(styled_df, use_container_width=True, height=400)

    time.sleep(POLLING_INTERVAL)
