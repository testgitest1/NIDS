
import os
import time
import subprocess
import pandas as pd
import numpy as np
import pickle
import threading
import socket
import struct
from datetime import datetime
from scapy.all import *
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class IntrusionDetectionSystem:
    def __init__(self, model_path, scaler_path, categorical_values_path):
        
        print("Initializing Intrusion Detection System...")
        # Create results directory
        if not os.path.exists('monitoring_results'):
            os.makedirs('monitoring_results')
        
        # Load the machine learning model
        self.load_model(model_path)
        
        # Load the scaler
        with open(scaler_path, 'rb') as file:
            self.scaler = pickle.load(file)
        
        # Load categorical values dictionary
        with open(categorical_values_path, 'rb') as file:
            self.categorical_values = pickle.load(file)
        
        # Define column names from NSL-KDD dataset for reference
        self.feature_names = self.get_feature_names()
        
        # Initialize logs
        self.log_file = 'monitoring_results/ids_detection_log.csv'
        self.initialize_log_file()
        
        # Network monitoring settings
        self.interface = None  # Will be set during setup
        self.capture_running = False
        self.flow_stats = {}
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 60  # Clean up flow stats every 60 seconds
        
        # For Scapy packet capture
        self.sniffer = None
    
    def load_model(self, model_path):
        print(f"Loading model from {model_path}...")
        try:
            # Try to load as a TensorFlow model (DNN or DQN)
            self.model = load_model(model_path)
            self.model_type = 'deep_learning'
        except:
            # Otherwise load as a scikit-learn model
            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)
            self.model_type = 'machine_learning'
        print(f"Model loaded successfully! Type: {self.model_type}")
    
    def get_feature_names(self):
        base_features = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate'
        ]
        return base_features
    
    def initialize_log_file(self):
        header = ['timestamp', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol',
                  'prediction', 'confidence', 'alert_level']
        
        # Create a new log file if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write(','.join(header) + '\n')
        print(f"Log file initialized at {self.log_file}")
    
    def log_detection(self, flow_id, features, prediction, confidence):
        # timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        src_ip, src_port, dst_ip, dst_port, protocol = flow_id
        
        # Determine alert level based on confidence
        if confidence > 0.9:
            alert_level = 'HIGH'
        elif confidence > 0.7:
            alert_level = 'MEDIUM'
        else:
            alert_level = 'LOW'
        
        # Log to file
        with open(self.log_file, 'a') as f:
            log_entry = [
                timestamp, src_ip, dst_ip, str(src_port), str(dst_port), protocol,
                'ATTACK' if prediction == 1 else 'NORMAL', 
                f"{confidence:.4f}", 
                alert_level
            ]
            f.write(','.join(log_entry) + '\n')
        
        # Print to console
        if prediction == 1:
            alert_color = '\033[91m' if alert_level == 'HIGH' else '\033[93m'  # Red for HIGH, Yellow for others
            print(f"{alert_color}[ALERT] {alert_level}: Potential attack detected!")
            print(f"Source: {src_ip}:{src_port} -> Destination: {dst_ip}:{dst_port} ({protocol})")
            print(f"Confidence: {confidence:.4f}\033[0m")  # Reset color
    
    def extract_features_from_packet_scapy(self, packet):
        """Extract relevant features from a Scapy packet and update flow statistics"""
        try:
            # Basic packet properties
            if IP in packet:
                src_ip = packet[IP].src
                dst_ip = packet[IP].dst
                
                # Determine protocol
                if TCP in packet:
                    src_port = packet[TCP].sport
                    dst_port = packet[TCP].dport
                    transport_proto = 'TCP'
                elif UDP in packet:
                    src_port = packet[UDP].sport
                    dst_port = packet[UDP].dport
                    transport_proto = 'UDP'
                else:
                    # Other IP protocols (ICMP, etc.)
                    src_port = 0
                    dst_port = 0
                    transport_proto = 'OTHER'
                
                # Create flow identifier (5-tuple)
                flow_id = (src_ip, src_port, dst_ip, dst_port, transport_proto)
                
                # Initialize flow stats if not exists
                if flow_id not in self.flow_stats:
                    self.flow_stats[flow_id] = {
                        'start_time': time.time(),
                        'last_seen': time.time(),
                        'packet_count': 0,
                        'byte_count': 0,
                        'src_bytes': 0,
                        'dst_bytes': 0,
                        'flags': set(),
                        'errors': 0,
                        'urgent': 0
                    }
                
                # Update flow statistics
                stats = self.flow_stats[flow_id]
                stats['last_seen'] = time.time()
                stats['packet_count'] += 1
                
                # Get packet length
                packet_len = len(packet)
                stats['byte_count'] += packet_len
                
                # Determine direction and update byte counts
                if stats['packet_count'] == 1:  # First packet defines forward direction
                    stats['src_bytes'] += packet_len
                else:
                    # Simple heuristic: if source matches the first packet's source, it's forward
                    if (src_ip, src_port) == (flow_id[0], flow_id[1]):
                        stats['src_bytes'] += packet_len
                    else:
                        stats['dst_bytes'] += packet_len
                
                # Check for TCP flags
                if TCP in packet:
                    flags_str = ""
                    tcp_flags = packet[TCP].flags
                    
                    # Convert flags to string representation
                    if tcp_flags & 0x01:  # FIN
                        flags_str += "F"
                    if tcp_flags & 0x02:  # SYN
                        flags_str += "S"
                    if tcp_flags & 0x04:  # RST
                        flags_str += "R"
                    if tcp_flags & 0x08:  # PSH
                        flags_str += "P"
                    if tcp_flags & 0x10:  # ACK
                        flags_str += "A"
                    if tcp_flags & 0x20:  # URG
                        flags_str += "U"
                    
                    stats['flags'].add(flags_str)
                    
                    # Check for urgent pointer
                    if tcp_flags & 0x20:  # URG flag
                        stats['urgent'] += 1
                
                # Return the flow ID for further processing
                return flow_id
            
        except Exception as e:
            print(f"Error processing packet: {e}")
        
        return None
    
    def prepare_flow_features(self, flow_id):
        """Convert flow statistics to NSL-KDD like features for model input"""
        # Get flow statistics
        stats = self.flow_stats[flow_id]
        
        # Calculate duration
        duration = stats['last_seen'] - stats['start_time']
        
        # Extract protocol type
        protocol_type = flow_id[4].lower()  # tcp, udp, or other
        
        # Determine service (port-based approximation)
        dst_port = flow_id[3]
        if dst_port == 80 or dst_port == 443:
            service = 'http'
        elif dst_port == 21:
            service = 'ftp'
        elif dst_port == 22:
            service = 'ssh'
        elif dst_port == 23:
            service = 'telnet'
        elif dst_port == 25:
            service = 'smtp'
        elif dst_port == 53:
            service = 'domain'
        else:
            service = 'other'
        
        # Determine flag (approximation based on TCP flags if available)
        if protocol_type == 'tcp' and stats['flags']:
            all_flags = ''.join(stats['flags'])
            if 'S' in all_flags and 'A' in all_flags:
                flag = 'S0'  # Connection established
            elif 'R' in all_flags:
                flag = 'REJ'  # Connection rejected
            elif 'F' in all_flags:
                flag = 'SF'  # Normal connection termination
            else:
                flag = 'S1'  # Connection in progress
        else:
            flag = 'OTH'  # Other
        
        # Prepare basic features dictionary
        features = {
            'duration': duration,
            'protocol_type': protocol_type,
            'service': service,
            'flag': flag,
            'src_bytes': stats['src_bytes'],
            'dst_bytes': stats['dst_bytes'],
            'land': 1 if flow_id[0] == flow_id[2] and flow_id[1] == flow_id[3] else 0,
            'wrong_fragment': 0,  # Simplified
            'urgent': stats['urgent'],
        }
        
        # Fill in other features with defaults or approximated values
        # Content features
        features.update({
            'hot': 0,  # Advanced analysis needed
            'num_failed_logins': 0,  # Would need session tracking
            'logged_in': 0,  # Would need session analysis
            'num_compromised': 0,  # Advanced analysis needed
            'root_shell': 0,  # Would need payload inspection
            'su_attempted': 0,  # Would need payload inspection
            'num_root': 0,  # Would need system monitoring
            'num_file_creations': 0,  # Would need system monitoring
            'num_shells': 0,  # Would need system monitoring
            'num_access_files': 0,  # Would need system monitoring
            'num_outbound_cmds': 0,  # Would need payload inspection
            'is_host_login': 0,  # Would need authentication monitoring
            'is_guest_login': 0,  # Would need authentication monitoring
        })
        
        # Time-based traffic features
        # These would require more sophisticated tracking across multiple flows
        conn_count = sum(1 for k, v in self.flow_stats.items() 
                         if k[2] == flow_id[2] and time.time() - v['last_seen'] < 2)
        srv_count = sum(1 for k, v in self.flow_stats.items() 
                        if k[2] == flow_id[2] and k[3] == flow_id[3] and time.time() - v['last_seen'] < 2)
        
        features.update({
            'count': conn_count,
            'srv_count': srv_count,
            'serror_rate': 0,  # Simplified
            'srv_serror_rate': 0,  # Simplified
            'rerror_rate': 0,  # Simplified
            'srv_rerror_rate': 0,  # Simplified
            'same_srv_rate': 1.0 if srv_count > 0 else 0,  # Simplified
            'diff_srv_rate': 0.0 if srv_count > 0 else 0,  # Simplified
            'srv_diff_host_rate': 0,  # Simplified
        })
        
        # Host-based traffic features
        dst_host_count = sum(1 for k, v in self.flow_stats.items() 
                            if k[2] == flow_id[2] and time.time() - v['last_seen'] < 120)
        dst_host_srv_count = sum(1 for k, v in self.flow_stats.items() 
                                if k[2] == flow_id[2] and k[3] == flow_id[3] and time.time() - v['last_seen'] < 120)
        
        features.update({
            'dst_host_count': min(dst_host_count, 255),  # Cap at 255 to match NSL-KDD scale
            'dst_host_srv_count': min(dst_host_srv_count, 255),  # Cap at 255
            'dst_host_same_srv_rate': 1.0 if dst_host_srv_count > 0 else 0,  # Simplified
            'dst_host_diff_srv_rate': 0.0 if dst_host_srv_count > 0 else 0,  # Simplified
            'dst_host_same_src_port_rate': 0,  # Simplified
            'dst_host_srv_diff_host_rate': 0,  # Simplified
            'dst_host_serror_rate': 0,  # Simplified
            'dst_host_srv_serror_rate': 0,  # Simplified
            'dst_host_rerror_rate': 0,  # Simplified
            'dst_host_srv_rerror_rate': 0,  # Simplified
        })
        
        return features
    
    def preprocess_features(self, features_dict):
        """Convert features dictionary to model input format"""
        # Create a DataFrame with the features
        df = pd.DataFrame([features_dict])
        
        # One-hot encode categorical features
        categorical_cols = ['protocol_type', 'service', 'flag']
        
        # Encode each categorical feature
        for col in categorical_cols:
            # Get the list of all possible values for this feature
            values = self.categorical_values[col]
            
            # Create columns for each value
            for val in values:
                df[f"{col}_{val}"] = (df[col] == val).astype(int)
            
            # Drop the original column
            df = df.drop(col, axis=1)
        
        # Get numerical columns (all except categorical)
        numerical_cols = [col for col in self.feature_names if col not in categorical_cols]
        
        # Scale numerical features
        df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        # Match columns with training data
        # This handles cases where some one-hot encoded columns might be missing
        model_columns = pd.read_csv('results/X_train.csv').columns
        
        # Create a new DataFrame with all model columns (fill missing with 0)
        model_input = pd.DataFrame(0, index=df.index, columns=model_columns)
        
        # Fill in values for columns that exist in our preprocessed data
        for col in df.columns:
            if col in model_columns:
                model_input[col] = df[col]
        
        return model_input
        
    def predict(self, flow_id):
        """Make a prediction for a flow based on extracted features"""
        # Prepare features for the flow
        raw_features = self.prepare_flow_features(flow_id)
        
        # Preprocess features to match model input format
        model_input = self.preprocess_features(raw_features)
        
        # Make prediction
        if self.model_type == 'deep_learning':
            # TensorFlow model
            prediction_prob = self.model.predict(model_input, verbose=0)[0][0]
            prediction = 1 if prediction_prob > 0.5 else 0
            confidence = prediction_prob if prediction == 1 else 1 - prediction_prob
        else:
            # Scikit-learn model
            prediction = self.model.predict(model_input)[0]
            
            # Get probability if the model supports it
            try:
                prediction_prob = self.model.predict_proba(model_input)[0]
                confidence = prediction_prob[1] if prediction == 1 else prediction_prob[0]
            except:
                # If predict_proba is not supported, use a fixed confidence
                confidence = 0.8 if prediction == 1 else 0.8
        
        # Log the detection
        self.log_detection(flow_id, raw_features, prediction, confidence)
        
        return prediction, confidence

    def start_live_monitoring(self, interface):
        """Start capturing and analyzing network traffic on the specified interface"""
        self.interface = interface
        self.capture_running = True
        print(f"\nStarting network monitoring on interface: {interface}")
        print("Press Ctrl+C to stop monitoring.\n")
        
        # Start the packet capture in a separate thread
        capture_thread = threading.Thread(target=self._capture_packets)
        capture_thread.daemon = True
        capture_thread.start()
        
        # Start the analysis thread
        analysis_thread = threading.Thread(target=self._analyze_flows)
        analysis_thread.daemon = True
        analysis_thread.start()
        
        try:
            # Keep the main thread running
            while self.capture_running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping network monitoring...")
            self.capture_running = False
            if self.sniffer:
                self.sniffer.stop()
            time.sleep(2)  # Give threads time to close
            print("Monitoring stopped.")

    def _packet_callback(self, packet):
        """Callback function for Scapy packet processing"""
        if not self.capture_running:
            return
            
        # Process the packet
        flow_id = self.extract_features_from_packet_scapy(packet)
        
        # Periodic cleanup of old flows
        current_time = time.time()
        if current_time - self.last_cleanup_time > self.cleanup_interval:
            self._cleanup_old_flows()
            self.last_cleanup_time = current_time

    def _capture_packets(self):
        """Capture packets using Scapy and process them"""
        try:
            print(f"Starting packet capture on interface: {self.interface}")
            # Use Scapy's sniff function for packet capture
            self.sniffer = AsyncSniffer(
                iface=self.interface,
                prn=self._packet_callback,
                store=False  # Don't store packets in memory
            )
            self.sniffer.start()
            
            # Keep the thread running until capture is stopped
            while self.capture_running:
                time.sleep(1)
                
        except Exception as e:
            print(f"Error in packet capture: {e}")
            self.capture_running = False

    def _analyze_flows(self):
        """Analyze flows periodically to detect potential intrusions"""
        try:
            while self.capture_running:
                # Sleep to reduce CPU usage
                time.sleep(2)
                
                # Get a list of flow IDs with enough packets for analysis
                flows_to_analyze = [
                    flow_id for flow_id, stats in self.flow_stats.items()
                    if stats['packet_count'] >= 3 and  # At least 3 packets
                    time.time() - stats['last_seen'] > 1 and  # Not too recent (collect more packets)
                    time.time() - stats['last_seen'] < 30  # Not too old (still relevant)
                ]
                
                # Analyze each flow
                for flow_id in flows_to_analyze:
                    try:
                        self.predict(flow_id)
                    except Exception as e:
                        print(f"Error analyzing flow {flow_id}: {e}")
                    
        except Exception as e:
            print(f"Error in flow analysis: {e}")
            self.capture_running = False

    def _cleanup_old_flows(self):
        """Remove old flows from memory to prevent memory leaks"""
        current_time = time.time()
        old_flows = [
            flow_id for flow_id, stats in self.flow_stats.items()
            if current_time - stats['last_seen'] > 300  # 5 minutes
        ]
        
        for flow_id in old_flows:
            del self.flow_stats[flow_id]
        
        if old_flows:
            print(f"Cleaned up {len(old_flows)} old flows. Current active flows: {len(self.flow_stats)}")

    def list_network_interfaces(self):
        """List available network interfaces"""
        try:
            # Use Scapy's built-in function to list interfaces
            interfaces = []
            if os.name == 'nt':  # Windows
                # On Windows, get interface names from Scapy
                from scapy.arch.windows import get_windows_if_list
                interfaces_info = get_windows_if_list()
                for interface in interfaces_info:
                    interfaces.append(f"{interface.get('name', 'Unknown')} ({interface.get('description', 'No description')})")
            else:  # Linux/Mac
                # On Linux/Mac, use the OS command for more user-friendly names
                try:
                    output = subprocess.check_output(['ifconfig'], stderr=subprocess.STDOUT).decode('utf-8')
                    lines = output.split('\n')
                    for line in lines:
                        if ':' in line and not line.startswith(' '):
                            interfaces.append(line.split(':')[0].strip())
                except:
                    # Fallback to Scapy's function
                    from scapy.arch import get_if_list
                    interfaces = get_if_list()
            
            return interfaces
        except Exception as e:
            print(f"Error listing network interfaces: {e}")
            # Fallback to basic Scapy interface list
            try:
                from scapy.arch import get_if_list
                return get_if_list()
            except:
                return ["eth0", "wlan0"]  # Default fallback interfaces
  
    def generate_report(self):
        """Generate a summary report of detected intrusions"""
        if not os.path.exists(self.log_file):
            print("No detections logged yet.")
            return
    
        try:
            # Read the log file
            df = pd.read_csv(self.log_file)
            
            # Generate summary statistics
            total_flows = len(df)
            
            # Check if we have any flows to analyze
            if total_flows == 0:
                print("\n===== Intrusion Detection Report =====")
                print("Total flows analyzed: 0")
                print("No network flows have been captured or analyzed yet.")
                print("Try running the monitoring for longer or on a more active network interface.")
                return
            
            attack_flows = df[df['prediction'] == 'ATTACK'].shape[0]
            normal_flows = total_flows - attack_flows
            
            # Alert level distribution
            high_alerts = df[df['alert_level'] == 'HIGH'].shape[0]
            medium_alerts = df[df['alert_level'] == 'MEDIUM'].shape[0]
            low_alerts = df[df['alert_level'] == 'LOW'].shape[0]
            
            # Top source IPs with attacks
            top_sources = df[df['prediction'] == 'ATTACK']['src_ip'].value_counts().head(5)
            
            # Top destination IPs under attack
            top_destinations = df[df['prediction'] == 'ATTACK']['dst_ip'].value_counts().head(5)
            
            # Print report
            print("\n===== Intrusion Detection Report =====")
            print(f"Total flows analyzed: {total_flows}")
            print(f"Normal flows: {normal_flows} ({normal_flows/total_flows*100:.1f}%)")
            print(f"Attack flows: {attack_flows} ({attack_flows/total_flows*100:.1f}%)")
            print("\nAlert Level Distribution:")
            print(f"  HIGH: {high_alerts} ({high_alerts/total_flows*100:.1f}%)")
            print(f"  MEDIUM: {medium_alerts} ({medium_alerts/total_flows*100:.1f}%)")
            print(f"  LOW: {low_alerts} ({low_alerts/total_flows*100:.1f}%)")
            
            print("\nTop Attack Sources:")
            for ip, count in top_sources.items():
                print(f"  {ip}: {count} potential attacks")
            
            print("\nTop Attack Targets:")
            for ip, count in top_destinations.items():
                print(f"  {ip}: {count} potential attacks")
            
            print("\nDetailed logs available at:", self.log_file)
        
        except Exception as e:
            print(f"Error generating report: {e}")
    


def main():
    # Model paths
    model_path = 'models/deep_learning/dnn_model.keras'  # Use your best performing model
    scaler_path = 'results/scaler.pkl'
    categorical_values_path = 'results/unique_categorical_values.pkl'
        
    # Initialize the IDS
    ids = IntrusionDetectionSystem(model_path, scaler_path, categorical_values_path)
    
    # Display welcome message and menu
    print("\n=== Real-time Network Intrusion Detection System ===")
    print("This system monitors network traffic and detects potential intrusions")
    print("using a machine learning model trained on the NSL-KDD dataset.")
    
    # List available interfaces
    available_interfaces = ids.list_network_interfaces()
    print("\nAvailable network interfaces:")
    for i, interface in enumerate(available_interfaces):
        print(f"{i+1}. {interface}")

    # subprocess.run(["snort", "-A", "console", "-i", interface, "-c", "snort.conf"])

    # Get user interface selection
    if available_interfaces:
        selection = int(input("\nSelect interface number to monitor: ")) - 1
        if 0 <= selection < len(available_interfaces):
            selected_interface = available_interfaces[selection]
            
            # Handle potential interface name format differences
            if '(' in selected_interface:  # Windows interface with description
                selected_interface = selected_interface.split('(')[0].strip()
            
            # Start monitoring
            ids.start_live_monitoring(selected_interface)
            
            # After monitoring stops, generate a report
            ids.generate_report()
        else:
            print("Invalid selection!")
    else:
        print("No network interfaces found!")


if __name__ == "__main__":
    main()