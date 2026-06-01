"""
Production Line OPCUA Server
A simulated production line with temperature, pressure, and speed sensors.
Sensors update their values every second.

Usage:
    python production_line_server.py
    
    Press Ctrl+C to stop the server.
"""

import time
import threading
import random
import sys
from opcua import ua, uamethod, Server


class ProductionLineSimulator:
    """Simulates a production line with three sensors."""
    
    def __init__(self):
        # Initialize sensor values with realistic ranges
        self.temperature = 25.0  # °C
        self.pressure = 5.0      # bar
        self.speed = 0.0         # RPM
        
        # Simulation parameters
        self.is_running = True
        self.lock = threading.Lock()
    
    def update_sensors(self):
        """Simulate gradual sensor value changes."""
        with self.lock:
            # Temperature: gradually fluctuate around 60°C with bounds 20-100°C
            temp_change = random.uniform(-2, 2)
            self.temperature = max(20, min(100, self.temperature + temp_change))
            
            # Pressure: gradually fluctuate around 5 bar with bounds 1-10 bar
            pressure_change = random.uniform(-0.5, 0.5)
            self.pressure = max(1, min(10, self.pressure + pressure_change))
            
            # Speed: gradually fluctuate around 500 RPM with bounds 0-1000 RPM
            # More likely to move toward 500 RPM (production speed)
            speed_target = 500
            speed_change = (speed_target - self.speed) * 0.1 + random.uniform(-50, 50)
            self.speed = max(0, min(1000, self.speed + speed_change))
    
    def get_values(self):
        """Get current sensor values."""
        with self.lock:
            return {
                'temperature': round(self.temperature, 2),
                'pressure': round(self.pressure, 2),
                'speed': round(self.speed, 2)
            }


def main():
    """Start the OPCUA server."""
    
    # Create simulator
    simulator = ProductionLineSimulator()
    
    # Create server
    server = Server()
    endpoint = "opc.tcp://127.0.0.1:4840"
    server.set_endpoint(endpoint)
    
    # Register namespace
    namespace_uri = "http://learning.opcua.production"
    namespace_idx = server.register_namespace(namespace_uri)
    
    print("[SERVER] OPCUA Production Line Server")
    print(f"[SERVER] Endpoint: {endpoint}")
    print(f"[SERVER] Namespace: {namespace_uri}")
    print()
    
    # Get root node
    root = server.get_objects_node()
    
    # Create ProductionLine folder
    production_line = root.add_folder(namespace_idx, "ProductionLine")
    print("[SERVER] Created folder: ProductionLine")
    
    # Create Sensors folder
    sensors_folder = production_line.add_folder(namespace_idx, "Sensors")
    print("[SERVER] Created folder: Sensors")
    
    # Create Temperature sensor
    temp_node = sensors_folder.add_variable(
        namespace_idx,
        "Temperature",
        ua.Variant(25.0, ua.VariantType.Float)
    )
    temp_node.set_writable()
    temp_unit = sensors_folder.add_variable(
        namespace_idx,
        "Temperature_Unit",
        ua.Variant("°C", ua.VariantType.String)
    )
    print("[SERVER] Created variable: Temperature (°C)")
    
    # Create Pressure sensor
    pressure_node = sensors_folder.add_variable(
        namespace_idx,
        "Pressure",
        ua.Variant(5.0, ua.VariantType.Float)
    )
    pressure_node.set_writable()
    pressure_unit = sensors_folder.add_variable(
        namespace_idx,
        "Pressure_Unit",
        ua.Variant("bar", ua.VariantType.String)
    )
    print("[SERVER] Created variable: Pressure (bar)")
    
    # Create Speed sensor
    speed_node = sensors_folder.add_variable(
        namespace_idx,
        "Speed",
        ua.Variant(0.0, ua.VariantType.Float)
    )
    speed_node.set_writable()
    speed_unit = sensors_folder.add_variable(
        namespace_idx,
        "Speed_Unit",
        ua.Variant("RPM", ua.VariantType.String)
    )
    print("[SERVER] Created variable: Speed (RPM)")
    print()
    
    # Start server
    server.start()
    print("[SERVER] Server started successfully!")
    print("[SERVER] Waiting for client connections...")
    print()
    
    def update_loop():
        """Background thread that updates sensor values every second."""
        counter = 0
        try:
            while simulator.is_running:
                time.sleep(1)
                
                # Simulate sensor updates
                simulator.update_sensors()
                values = simulator.get_values()
                
                # Update OPCUA nodes
                temp_node.set_value(values['temperature'])
                pressure_node.set_value(values['pressure'])
                speed_node.set_value(values['speed'])
                
                counter += 1
                print(f"[SERVER] Update #{counter}: T={values['temperature']:.1f}°C, "
                      f"P={values['pressure']:.1f}bar, S={values['speed']:.0f}RPM")
        
        except Exception as e:
            print(f"[SERVER] Error in update loop: {e}")
        finally:
            print("[SERVER] Update loop stopped")
    
    # Start background update thread
    update_thread = threading.Thread(target=update_loop, daemon=True)
    update_thread.start()
    
    # Keep server running
    try:
        while True:
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print()
        print("[SERVER] Shutdown requested...")
    
    finally:
        simulator.is_running = False
        server.stop()
        print("[SERVER] Server stopped")


if __name__ == "__main__":
    main()
