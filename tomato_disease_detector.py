import cv2
import numpy as np
import time
from djitellopy import Tello
import threading

class TomatoDiseaseDetector:
    def __init__(self):
        print("Inițializare dronă Tello...")
        self.tello = Tello()
        
        # Verificare conectivitate
        if not self._connect_to_drone():
            raise ConnectionError("Nu s-a putut conecta la dronă. Verifică Wi-Fi-ul și starea dronei.")
        
        # Configurații pentru performanță video
        self.tello.set_video_fps(Tello.FPS_30)  # 30 FPS
        self.tello.set_video_bitrate(Tello.BITRATE_5MBPS)  # Bitrate maxim
        self.tello.set_video_resolution(Tello.RESOLUTION_720P)  # Rezoluție 720p
        
        # Pornire flux video
        print("Pornire flux video...")
        self.tello.streamon()
        time.sleep(2)  # Așteaptă stabilizarea fluxului
        
        # Parametri pentru controlul dronei
        self.flight_speed = 20  # Viteză implicită
        self.active_command = None  # Comanda activă curentă
        self.command_thread = None  # Thread pentru comenzi continue
        self.running = True  # Flag pentru controlul thread-urilor
        
        # Parametri pentru procesarea imaginii
        self.last_frame = None  # Ultimul cadru procesat
    
    def _connect_to_drone(self):
        try:
            self.tello.connect()
            print(f"Nivel baterie: {self.tello.get_battery()}%")
            if self.tello.get_battery() < 10:
                print("Avertisment: Bateria dronei este sub 10%. Încarcă bateria!")
            return True
        except Exception as e:
            print(f"Eroare la conectarea cu drona: {e}")
            return False
    
    def maintain_altitude(self):
        self.tello.send_rc_control(0, 0, 3, 0)  # Viteză verticală mică
    
    def takeoff(self):
        print("Inițializare dronă...")
        self.tello.send_rc_control(0, 0, 0, 0)  # Resetează comenzile RC
        time.sleep(1)
        print("Decolare...")
        self.tello.takeoff()
        time.sleep(2)
        print("Ridicare la altitudine stabilă...")
        self.tello.send_rc_control(0, 0, 20, 0)  # Urcă cu viteză 20
        time.sleep(2)
        self.maintain_altitude()
    
    def land(self):
        print("Aterizare...")
        self.stop_continuous_command()
        try:
            self.tello.land()
        except Exception as e:
            print(f"Eroare la aterizare: {e}")
    
    def _continuous_command(self, command_func):
        """
        Execută o comandă continuu până la oprire.
        Args:
            command_func: Funcția de comandă de executat.
        """
        while self.running and self.active_command:
            command_func()
            time.sleep(0.03)  # Frecvență mai mare pentru control fluid
    
    def start_continuous_command(self, command_type):
        """
        Pornește o comandă care să ruleze continuu în background.
        Args:
            command_type (str): Tipul comenzii ('forward', 'backward', etc.).
        """
        self.stop_continuous_command()
        self.active_command = command_type
        
        command_funcs = {
            'forward': lambda: self.tello.send_rc_control(0, self.flight_speed, 0, 0),
            'backward': lambda: self.tello.send_rc_control(0, -self.flight_speed, 0, 0),
            'left': lambda: self.tello.send_rc_control(-self.flight_speed, 0, 0, 0),
            'right': lambda: self.tello.send_rc_control(self.flight_speed, 0, 0, 0),
            'up': lambda: self.tello.send_rc_control(0, 0, self.flight_speed, 0),
            'down': lambda: self.tello.send_rc_control(0, 0, -self.flight_speed//2, 0),
            'rotate_cw': lambda: self.tello.send_rc_control(0, 0, 0, self.flight_speed),
            'rotate_ccw': lambda: self.tello.send_rc_control(0, 0, 0, -self.flight_speed)
        }
        
        if command_type in command_funcs:
            self.command_thread = threading.Thread(
                target=self._continuous_command,
                args=(command_funcs[command_type],)
            )
            self.command_thread.daemon = True
            self.command_thread.start()
    
    def stop_continuous_command(self):
        """Oprește orice comandă continuă activă"""
        self.active_command = None
        if self.command_thread and self.command_thread.is_alive():
            self.command_thread.join(timeout=0.5)
        self.tello.send_rc_control(0, 0, 0, 0)
        self.maintain_altitude()
    
    def move_forward(self):
        print("Drona se mișcă înainte...")
        self.start_continuous_command('forward')
    
    def move_backward(self):
        print("Drona se mișcă înapoi...")
        self.start_continuous_command('backward')
    
    def move_left(self):
        print("Drona se mișcă la stânga...")
        self.start_continuous_command('left')
    
    def move_right(self):
        print("Drona se mișcă la dreapta...")
        self.start_continuous_command('right')
    
    def move_up(self):
        print("Drona se mișcă în sus...")
        self.start_continuous_command('up')
    
    def move_down(self):
        print("Drona se mișcă în jos...")
        self.start_continuous_command('down')
    
    def rotate_clockwise(self):
        print("Drona se rotește în sensul acelor de ceasornic...")
        self.start_continuous_command('rotate_cw')
    
    def rotate_counterclockwise(self):
        print("Drona se rotește în sens invers acelor de ceasornic...")
        self.start_continuous_command('rotate_ccw')
    
    def increase_speed(self):
        self.flight_speed = min(self.flight_speed + 5, 50)
        print(f"Viteză mărită la: {self.flight_speed}")
    
    def decrease_speed(self):
        self.flight_speed = max(self.flight_speed - 5, 10)
        print(f"Viteză redusă la: {self.flight_speed}")
    
    def capture_frame(self):
        frame = self.tello.get_frame_read().frame
        if frame is None:
            if self.last_frame is not None:
                return self.last_frame
            return np.zeros((720, 960, 3), dtype=np.uint8)
        
        # Procesare minimă pentru a crește FPS-ul
        alpha = 1.1  # Contrast minim
        beta = 5     # Luminozitate minimă
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        
        self.last_frame = frame
        return frame
    
    def run_detection(self):

        print("Comenzi disponibile:")
        print("  t: Decolare (ridică drona la ~1 metru)")
        print("  l: Aterizare (oprește drona și o coboară)")
        print("  [Taste menținute apăsate pentru mișcare continuă]")
        print("  w: Înainte")
        print("  s: Înapoi")
        print("  a: Stânga")
        print("  d: Dreapta")
        print("  e: Sus")
        print("  z: Jos (coboară treptat)")
        print("  r: Rotire dreapta")
        print("  y: Rotire stânga")
        print("  spațiu: Stop (oprește orice mișcare)")
        print("  +: Mărește viteza")
        print("  -: Reduce viteza")
        print("  q: Ieșire")
        
        cv2.namedWindow("Live Stream", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Live Stream", 1280, 720)
        
        last_command = "Așteaptă o comandă..."
        key_pressed = False
        
        try:
            while self.running:
                frame = self.capture_frame()
                
                # Afișare status
                status_text = [
                    f"Comandă: {last_command}",
                    f"Viteză: {self.flight_speed}",
                    f"Baterie: {self.tello.get_battery()}%",
                    f"Mișcare activă: {self.active_command if self.active_command else 'Niciuna'}"
                ]
                for i, text in enumerate(status_text):
                    cv2.putText(frame, text, (10, 30 + i*30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                cv2.imshow("Live Stream", frame)
                
                key = cv2.waitKey(5) & 0xFF
                
                if key == ord('q'):
                    last_command = "Ieșire"
                    break
                elif key == ord('t'):
                    last_command = "Decolare"
                    self.takeoff()
                elif key == ord('l'):
                    last_command = "Aterizare"
                    self.land()
                elif key == ord(' '):
                    last_command = "Stop"
                    self.stop_continuous_command()
                elif key == ord('w'):
                    last_command = "Înainte"
                    self.move_forward()
                    key_pressed = True
                elif key == ord('s'):
                    last_command = "Înapoi"
                    self.move_backward()
                    key_pressed = True
                elif key == ord('a'):
                    last_command = "Stânga"
                    self.move_left()
                    key_pressed = True
                elif key == ord('d'):
                    last_command = "Dreapta"
                    self.move_right()
                    key_pressed = True
                elif key == ord('e'):
                    last_command = "Sus"
                    self.move_up()
                    key_pressed = True
                elif key == ord('z'):
                    last_command = "Jos"
                    self.move_down()
                    key_pressed = True
                elif key == ord('r'):
                    last_command = "Rotire dreapta"
                    self.rotate_clockwise()
                    key_pressed = True
                elif key == ord('y'):
                    last_command = "Rotire stânga"
                    self.rotate_counterclockwise()
                    key_pressed = True
                elif key == ord('+') or key == ord('='):
                    last_command = "Mărire viteză"
                    self.increase_speed()
                elif key == ord('-') or key == ord('_'):
                    last_command = "Reducere viteză"
                    self.decrease_speed()
                elif key == 255 and key_pressed:
                    self.stop_continuous_command()
                    key_pressed = False
                    
        except Exception as e:
            print(f"Eroare în timpul rulării: {e}")
            
        finally:
            self.running = False
            if self.command_thread and self.command_thread.is_alive():
                self.command_thread.join(timeout=1.0)
            
            try:
                if self.tello.get_height() > 10:
                    print("Aterizare automată pentru siguranță...")
                    self.land()
            except Exception as e:
                print(f"Eroare la verificarea înălțimii: {e}")
            
            self.tello.streamoff()
            cv2.destroyAllWindows()
    
    def close(self):
        try:
            self.tello.end()
        except Exception as e:
            print(f"Eroare la închiderea conexiunii: {e}")

def main():
    try:
        detector = TomatoDiseaseDetector()
        print("Apasă Enter pentru a începe...")
        input()
        detector.run_detection()
    except Exception as e:
        print(f"Eroare la inițializare: {e}")
    finally:
        if 'detector' in locals():
            detector.close()

if __name__ == "__main__":
    main()