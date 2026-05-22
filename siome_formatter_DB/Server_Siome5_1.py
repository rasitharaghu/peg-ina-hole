import random
import time
import logging
from datetime import datetime

try:
    from opcua import Server
except Exception:
    raise RuntimeError("python-opcua is required. Install with: pip install opcua")


def create_demo_server(endpoint="opc.tcp://0.0.0.0:4840"):
    server = Server()
    server.set_endpoint(endpoint)

    uri = "http://examples.freeopcua.github.io"
    idx = server.register_namespace(uri)

    objects = server.get_objects_node()
    factory_obj = objects.add_object(idx, "ProductionLine_01")

    variables = {
        "Temperature": factory_obj.add_variable(idx, "Temperature", 70.0),
        "Pressure": factory_obj.add_variable(idx, "Pressure", 6.0),
        "Torque": factory_obj.add_variable(idx, "Torque", 11.0),
        "MotorRotation": factory_obj.add_variable(idx, "MotorRotation", 1500),
        "Status": factory_obj.add_variable(idx, "Status", "Idle"),
        "FaultCode": factory_obj.add_variable(idx, "FaultCode", 0),

        # Extra robot/process-style tags
        "CycleTime": factory_obj.add_variable(idx, "CycleTime", 0.0),
        "PartCount": factory_obj.add_variable(idx, "PartCount", 0),
        "ToolWear": factory_obj.add_variable(idx, "ToolWear", 0.0),
        "Vibration": factory_obj.add_variable(idx, "Vibration", 0.0),
        "Current": factory_obj.add_variable(idx, "Current", 0.0),
        "Voltage": factory_obj.add_variable(idx, "Voltage", 230.0),
        "Power": factory_obj.add_variable(idx, "Power", 0.0),
        "WorkOrder": factory_obj.add_variable(idx, "WorkOrder", "WO_001"),
        "TaskIdentifier": factory_obj.add_variable(idx, "TaskIdentifier", "TASK_001"),
        "ResultEvaluation": factory_obj.add_variable(idx, "ResultEvaluation", 1),
    }

    for var in variables.values():
        var.set_writable()

    return server, variables


def run_demo_loop(server, variables, update_interval=1.0):
    logging.info("Starting Industrial Simulation loop. Press Ctrl+C to stop.")

    current_temp = 70.0
    part_count = 0
    t = 0

    try:
        while True:
            current_temp += random.uniform(-0.5, 0.5)
            current_pressure = round(random.uniform(5.8, 6.2), 2)
            current_torque = round(random.uniform(10.5, 12.0), 2)
            current_rpm = random.randint(1450, 1550)
            current_status = "Running" if current_rpm > 0 else "Stopped"
            current_fault = 0 if random.random() > 0.01 else 404

            cycle_time = round(random.uniform(8.0, 12.0), 2)
            tool_wear = round(min(100.0, t * 0.05 + random.uniform(0, 0.2)), 2)
            vibration = round(random.uniform(0.1, 1.5), 3)
            current = round(random.uniform(4.5, 7.5), 2)
            voltage = round(random.uniform(225.0, 235.0), 2)
            power = round(current * voltage, 2)

            if t % 10 == 0:
                part_count += 1

            variables["Temperature"].set_value(round(current_temp, 2))
            variables["Pressure"].set_value(current_pressure)
            variables["Torque"].set_value(current_torque)
            variables["MotorRotation"].set_value(current_rpm)
            variables["Status"].set_value(current_status)
            variables["FaultCode"].set_value(current_fault)

            variables["CycleTime"].set_value(cycle_time)
            variables["PartCount"].set_value(part_count)
            variables["ToolWear"].set_value(tool_wear)
            variables["Vibration"].set_value(vibration)
            variables["Current"].set_value(current)
            variables["Voltage"].set_value(voltage)
            variables["Power"].set_value(power)
            variables["WorkOrder"].set_value(f"WO_{1000 + part_count}")
            variables["TaskIdentifier"].set_value(f"TASK_{part_count}")
            variables["ResultEvaluation"].set_value(0 if current_fault else 1)

            logging.info(
                "Update -> Temp: %.2f | Pressure: %.2f | Torque: %.2f | RPM: %s | Status: %s | Fault: %s",
                current_temp,
                current_pressure,
                current_torque,
                current_rpm,
                current_status,
                current_fault,
            )

            t += 1
            time.sleep(update_interval)

    except KeyboardInterrupt:
        logging.info("Demo loop interrupted by user")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )

    endpoint = "opc.tcp://0.0.0.0:4840"
    server, variables = create_demo_server(endpoint=endpoint)

    try:
        server.start()
        logging.info("Industrial OPC UA server started at %s", endpoint)
        logging.info("Nodes created: %s", list(variables.keys()))

        run_demo_loop(server, variables, update_interval=1.0)

    finally:
        logging.info("Stopping OPC UA server...")
        server.stop()


if __name__ == "__main__":
    main()