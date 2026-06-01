# OPCUA Production Line - Learning Project

A hands-on learning project demonstrating OPCUA (OPC Unified Architecture) concepts using a simulated production line with real-time sensor monitoring.

## Project Overview

This project implements a complete OPCUA client-server architecture:

- **Server**: Simulates a production line with three sensors (Temperature, Pressure, Speed) that update every second
- **Client**: Demonstrates core OPCUA operations: Connect → Browse → Read → Subscribe

### Architecture

```
Production Line Server (localhost:4840)
├── ProductionLine (Folder)
│   └── Sensors (Folder)
│       ├── Temperature (Float) + Unit (String)
│       ├── Pressure (Float) + Unit (String)
│       └── Speed (Float) + Unit (String)
```

## Prerequisites

- Python 3.7+
- Python OPCUA library (already in your `opcua_env` environment)

## Project Structure

```
LearnOPCUA/
├── server/
│   └── production_line_server.py    # OPCUA server with simulation
├── client/
│   └── production_line_client.py    # OPCUA client (connect, browse, read, subscribe)
└── README.md                         # This file
```

## Setup

### 1. Activate Python Environment

```powershell
# Activate your OPCUA environment
conda activate opcua_env
```

### 2. Navigate to Project Directory

```powershell
cd c:\Rasitha\LearnOPCUA
```

## Running the Project

You'll need **two terminal windows**: one for the server and one for the client.

### Terminal 1: Start the Server

```powershell
cd c:\Rasitha\LearnOPCUA
conda activate opcua_env
python server/production_line_server.py
```

**Expected Output:**
```
[SERVER] OPCUA Production Line Server
[SERVER] Endpoint: opc.tcp://127.0.0.1:4840
[SERVER] Namespace: http://learning.opcua.production

[SERVER] Created folder: ProductionLine
[SERVER] Created folder: Sensors
[SERVER] Created variable: Temperature (°C)
[SERVER] Created variable: Pressure (bar)
[SERVER] Created variable: Speed (RPM)

[SERVER] Server started successfully!
[SERVER] Waiting for client connections...

[SERVER] Update #1: T=25.0°C, P=5.0bar, S=50.0RPM
[SERVER] Update #2: T=26.5°C, P=5.2bar, S=120.3RPM
...
```

The server will keep updating sensor values. Leave it running.

### Terminal 2: Run the Client

```powershell
cd c:\Rasitha\LearnOPCUA
conda activate opcua_env
python client/production_line_client.py
```

**Expected Output:**

```
[CLIENT] Production Line OPCUA Client
[CLIENT] Endpoint: opc.tcp://127.0.0.1:4840

[CLIENT] Step 1: CONNECT
--------------------------------------------------
[CLIENT] Connected successfully!

[CLIENT] Step 2: BROWSE
--------------------------------------------------
[CLIENT] Namespace structure:
  ├─ Root (ID: i=84)
  ├─ Objects (ID: i=85)
    ├─ ProductionLine (ID: ns=1;s=ProductionLine)
      ├─ Sensors (ID: ns=1;s=Sensors)
        ├─ Temperature (ID: ns=1;s=Temperature)
        ├─ Temperature_Unit (ID: ns=1;s=Temperature_Unit)
        ├─ Pressure (ID: ns=1;s=Pressure)
        ├─ Pressure_Unit (ID: ns=1;s=Pressure_Unit)
        ├─ Speed (ID: ns=1;s=Speed)
        ├─ Speed_Unit (ID: ns=1;s=Speed_Unit)
  ...

[CLIENT] Step 3: READ
--------------------------------------------------
Reading sensor values...
--------------------------------------------------
  Temperature: 27.3 °C
  Pressure: 5.1 bar
  Speed: 245.6 RPM
--------------------------------------------------

[CLIENT] Step 4: SUBSCRIBE
--------------------------------------------------
[CLIENT] Subscribing to changes for 15 seconds...
--------------------------------------------------
[CLIENT] Subscribed to: Temperature
[CLIENT] Subscribed to: Pressure
[CLIENT] Subscribed to: Speed
--------------------------------------------------
[CLIENT] Listening for updates...

[14:32:15] Temperature changed to: 28.1
[14:32:16] Pressure changed to: 5.3
[14:32:16] Speed changed to: 312.4
[14:32:17] Temperature changed to: 28.9
[14:32:18] Speed changed to: 380.1
...

[CLIENT] Subscription completed
[CLIENT] Disconnected successfully
```

## Workflow Explanation

### 1. **CONNECT** (Server Ready State)
- Client initializes connection to `opc.tcp://127.0.0.1:4840`
- Establishes secure/unsecure channel
- Authentication (in this case, anonymous)

### 2. **BROWSE** (Discover Namespace)
- Client recursively traverses the namespace tree
- Displays all nodes, folders, and variables
- Shows node IDs (important for direct node access)
- Learns the structure without reading values

### 3. **READ** (Get Current Values)
- Client queries current sensor values from server
- Single-shot read operation (no subscription)
- Gets Temperature, Pressure, Speed with units
- Useful for one-time data retrieval

### 4. **SUBSCRIBE** (Real-time Monitoring)
- Client creates a subscription with 1000ms update rate
- Adds monitored items for each sensor
- Server sends data change notifications automatically
- Client callback handler prints updates in real-time
- Runs for 15 seconds to demonstrate continuous monitoring

## Key Concepts Demonstrated

| Concept | Where | Purpose |
|---------|-------|---------|
| **Namespace** | Server creates custom namespace URI | Organize nodes logically |
| **Nodes** | Folders and Variables | Structure the server's information model |
| **Variables** | Temperature, Pressure, Speed | Store readable/writable values |
| **Threading** | Server update loop | Background simulation while accepting clients |
| **Connection** | Client.connect() | Establish OPCUA channel |
| **Browsing** | get_children() recursion | Discover server structure |
| **Reading** | node.get_value() | Synchronous data retrieval |
| **Subscription** | create_subscription() | Asynchronous change notifications |
| **Data Binding** | SubscriptionHandler | Callback-based event handling |

## Sensor Ranges and Behavior

- **Temperature**: 20–100°C, fluctuates ±2°C per second
- **Pressure**: 1–10 bar, fluctuates ±0.5 bar per second
- **Speed**: 0–1000 RPM, gradually trends toward 500 RPM with random variance

This simulates realistic production line behavior with sensor noise and gradual state transitions.

## Stopping the Project

- **Server**: Press `Ctrl+C` (gracefully shuts down)
- **Client**: Completes automatically after 15 seconds or press `Ctrl+C`

Both clean up resources properly and disconnect.

## Troubleshooting

### Client can't connect to server
- ✓ Verify server is running in first terminal
- ✓ Check endpoint: `opc.tcp://127.0.0.1:4840`
- ✓ Ensure both are using the same port (4840)

### No sensor updates visible
- ✓ Check server terminal for update messages
- ✓ Verify subscription is created (look for "Subscribed to" messages)
- ✓ Wait for server's update cycle (every 1 second)

### Import errors (opcua module not found)
- ✓ Ensure `opcua_env` is activated: `conda activate opcua_env`
- ✓ Verify python-opcua is installed: `pip list | grep opcua`

## Next Steps for Learning

1. **Modify sensor ranges**: Edit simulator parameters in `production_line_server.py`
2. **Add more sensors**: Duplicate sensor creation code with new variables
3. **Write values**: Use `node.set_value()` to control server from client
4. **Add security**: Implement username/password authentication
5. **Multiple clients**: Run multiple client instances simultaneously
6. **Persistence**: Save sensor history to CSV or database
7. **Web UI**: Build a web dashboard using the sensor data

## References

- [OPC Unified Architecture (Wikipedia)](https://en.wikipedia.org/wiki/OPC_Unified_Architecture)
- [python-opcua Documentation](https://python-opcua.readthedocs.io/)
- [OPC Foundation](https://opcfoundation.org/)

## Notes

- This is a learning project. For production use, add proper error handling, logging, security, and monitoring.
- The server simulates gradual sensor changes for realism, not pure randomness.
- Subscription update rate is set to 1000ms (1 second) but values update every 1 second, so you'll see changes frequently.
