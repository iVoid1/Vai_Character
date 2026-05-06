from pythonosc import udp_client
import time
import math
import struct
import json

class VMCController:
    def __init__(self, ip="127.0.0.1", port=8000, vrm_path=None):
        self.client = udp_client.SimpleUDPClient(ip, port)
        self.current_angles = {"Head": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}}
        self.available_expressions = self.get_vrm_blendshapes(vrm_path) if vrm_path else None
        self.expressions = {expr: 0.0 for expr in self.available_expressions} if self.available_expressions else {}
        self.set_expression(self.expressions, normalize=False)  # Initialize all expressions to 0
        
    def get_vrm_blendshapes(self, file_path):
        with open(file_path, 'rb') as f:
            f.read(12)  # Skip header
            chunk_length = struct.unpack('<I', f.read(4))[0]
            f.read(4)  # Skip type
            json_data = json.loads(f.read(chunk_length))
            
        # 1. Detect Version & Extract Expressions
        extensions = json_data.get("extensions", {})
        
        # Check for VRM 1.0 (VRMC_vrm)
        if "VRMC_vrm" in extensions:
            vrm_data = extensions["VRMC_vrm"]
            # Structure: extensions -> VRMC_vrm -> expressions -> preset
            expressions = vrm_data.get("expressions", {}).get("preset", {})
            return list(expressions.keys())
        
        # Check for VRM 0.x (VRM)
        elif "VRM" in extensions:
            vrm_data = extensions["VRM"]
            # Structure: extensions -> VRM -> blendShapeMaster -> blendShapeGroups
            groups = vrm_data.get("blendShapeMaster", {}).get("blendShapeGroups", [])
            return [g.get("name") for g in groups if g.get("name")]

        return []

    def set_expression(self, blendshape_dict, normalize=True):
        """Sends expression values (0.0 - 100.0)"""
        for name, value in blendshape_dict.items():
            # Ensure float conversion and range normalization
            if normalize:
                value = float(value / 100)
            self.client.send_message("/VMC/Ext/Blend/Val", [name, value])
        self.client.send_message("/VMC/Ext/Blend/Apply", [])
        
    def smooth_expression(self, blendshape_dict, normalize=True, smoothness=0.1):
        """Smoothly transitions current expression values toward target values"""
        for expression_name, target_value in blendshape_dict.items():
            self.expressions[expression_name] = self.expressions.get(expression_name, 0.) + (target_value - self.expressions.get(expression_name, 0.)) * smoothness
            self.set_expression({expression_name: self.expressions[expression_name]}, normalize=normalize)
        
    def _euler_to_quaternion(self, pitch, yaw, roll):
        p = math.radians(pitch)
        y = math.radians(yaw)
        r = math.radians(roll)

        cy = math.cos(y * 0.5)
        sy = math.sin(y * 0.5)
        cp = math.cos(p * 0.5)
        sp = math.sin(p * 0.5)
        cr = math.cos(r * 0.5)
        sr = math.sin(r * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        return [qx, qy, qz, qw]

    def update_bone(self, bone_name, pitch, yaw, roll, x=0.0, y=0.0, z=0.0):
        """Calculates and sends bone transformation"""
        q = self._euler_to_quaternion(pitch, yaw, roll)
        payload = [str(bone_name), float(x), float(y), float(z), 
                   float(q[0]), float(q[1]), float(q[2]), float(q[3])]
        self.client.send_message("/VMC/Ext/Bone/Pos", payload)

    def smooth_move(self, bone_name, target_angles, lerp_speed=0.1):
        """Interpolates current angles toward target angles for fluid motion"""
        for axis in ["pitch", "yaw", "roll"]:
            current = self.current_angles[bone_name][axis]
            target = target_angles.get(axis, 0)
            # Linear interpolation (Lerp)
            self.current_angles[bone_name][axis] += (target - current) * lerp_speed
        
        angles = self.current_angles[bone_name]
        self.update_bone(bone_name, angles["pitch"], angles["yaw"], angles["roll"])

if __name__ == "__main__":
    vmc = VMCController("127.0.0.1", 8000, r"C:\Users\Void\Documents\AvatarSample_I.vrm")
    
    # Optional: Load blendshapes to verify
    # blends = vmc.get_vrm_blendshapes(vrm_path)
    
    tick = 0
    # print(vmc.available_expressions)  # Print available blendshapes for reference
    try:
        while True:
            tick += 0.05  # Increment time for animation
            vmc.smooth_move(
                bone_name="Head", 
                target_angles={
                    "pitch": math.sin(tick) * 15.0,
                    "yaw": math.sin(tick * 0.3) * 15.0,
                    "roll": math.cos(tick * 0.5) * 15.0
                    },
                lerp_speed=0.1
                )
            vmc.smooth_expression({
                vmc.available_expressions[0]: 50,  # Example: oscillate first blendshape
                # vmc.available_expressions[1]: 3,
                # vmc.available_expressions[4]: 30,
                # vmc.available_expressions[3]: 50,
                vmc.available_expressions[11]:-25,
                vmc.available_expressions[12]: 50
            }, normalize=True, smoothness=0.05)
            # Maintain 60Hz update rate for maximum smoothness
            tick = 0 if tick > 100000 else tick
            time.sleep(1/120)
    except KeyboardInterrupt:
        print("VMC Controller stopped.")