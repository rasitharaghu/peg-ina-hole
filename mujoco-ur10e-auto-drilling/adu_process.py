from bt import Status

class ADUProcessController:
    def __init__(self):
        self.spindle_on = False
        self.feed_enabled = False
        self.depth_reached = False

    def start_spindle(self):
        self.spindle_on = True
        print("[ADU] spindle ON")

    def stop_spindle(self):
        self.spindle_on = False
        print("[ADU] spindle OFF")

    def enable_feed(self):
        self.feed_enabled = True
        print("[ADU] feed ENABLED")

    def disable_feed(self):
        self.feed_enabled = False
        print("[ADU] feed DISABLED")

    def mark_depth_reached(self):
        self.depth_reached = True
        print("[ADU] depth reached")

class ADUStartAction:
    def __init__(self, adu):
        self.adu = adu
        self.done = False

    def tick(self):
        if not self.done:
            self.adu.start_spindle()
            self.adu.enable_feed()
            self.done = True
        return Status.SUCCESS

class ADUStopAction:
    def __init__(self, adu):
        self.adu = adu
        self.done = False

    def tick(self):
        if not self.done:
            self.adu.disable_feed()
            self.adu.stop_spindle()
            self.done = True
        return Status.SUCCESS
