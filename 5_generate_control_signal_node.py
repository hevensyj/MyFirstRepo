import rclpy
import math
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge 
from message_filters import TimeSynchronizer, ApproximateTimeSynchronizer, Subscriber
from interfaces_pkg.msg import LaneInfo, Motor

'''
도로 영상의 정보들을 이용하여 최종 조향 단계를 추출한 노드
---------------------------------------------------------------------------

<받는 데이터>                                                                  
  road_gradient : 주행 도로의 기울기
  road_target_point_x : 주행 도로의 목표점의 x 좌표
  road_target_point_y : 주행 도로의 목표점의 y 좌표

<변수>
  left_speed_command, right_speed_command : 좌, 우 뒷바퀴 속도
  car_center_point_x, car_center_point_y : 차량 앞 범퍼의 중심이 위치한 픽셀의 좌표
  target_slope : 주행 도로의 목표점의 좌표와 차량 앞 범퍼의 중심이 위치한 좌표 사이의 기울기
  
  steering_command[(-7) ~ (+7)]: 
    음수는 좌측 조향, 양수는 우측 조향, 0은 직진
    숫자의 절댓값이 클 수록 조향각이 커짐. 
    
  actual_left_speed_command[(-255) ~ (+255)]:
    좌측뒷바퀴 모터의 속도
    음수는 후진, 양수는 전진
    숫자의 절댓값이 클 수록 모터가 빠르게 회전
    
  actual_right_speed_command[(-255) ~ (+255)]:
    우측뒷바퀴 모터의 속도
    음수는 후진, 양수는 전진
    숫자의 절댓값이 클 수록 모터가 빠르게 회전

* update_actual_speed 함수는 속도가 급격하게 변하여 모터가 고장나는 것을 막기위함.

* road_gradient와 target_slope를 활용하여 자유롭게 알고리즘 작성하여 
steering_command, actual_left_speed_command, actual_right_speed_command 를 도출하시오.

---------------------------------------------------------------------------
'''

from .lib import control_motor as CONTROL

#---TODO-------------------------------------
SUB_TOPIC_NAME = "topic_lane_info"
PUB_TOPIC_NAME = "topic_contol_signal"

# 차량의 속도 (-255 ~ +255) , 음수면 후진, 양수면 전진
SPEED = 50

# 급격한 모터 속도 변화 완화(클수록 더욱 부드럽게 속도 변화)
SPEED_CHANGE_SMOOTHNESS = 30

# 차량 앞 범퍼의 중심이 위치한 픽셀의 좌표 (????). 수정 필요
car_center_point_x = 280
car_center_point_y = 179

#=> 차량의 방향을 grad로 구한 후 전륜축 구할 때 참고하기 (일단 보류)

# 카메라상 이미지와 실제 이미지 비율 (p는 이미지의 실제 가로 길이, q는 이미지의 실제 세로 길이)
p = 0.85 *  640 / 300 # 도로 픽셀: 300, 실제 도로 폭: 0.85. 단위: m
q = 2 # 수정 필요


# 차량의 전륜축으로부터 범퍼 앞쪽이 떨어진 위치 (사실상 D_1 = D_2)
D_1 = 1 # 수정 필요. 단위: m

# 차량의 전륜축으로부터 범퍼 앞쪽이 떨어진 위치 (사실상 D_1 = D_2)
D_2 = 1 # 수정 필요. 단위: m

# 차량의 전륜축으로부터 카메라가 담는 실제 이미지 영역이 떨어진 위치
D_3 = 1 # 수정 필요. 단위: m

TIMER = 0.1
QUE = 1
#--------------------------------------------

class IntegrateInfo():
  def __init__(self):
    self.smoothness = SPEED_CHANGE_SMOOTHNESS
    self.left_smooth_speed_controller = CONTROL.smoooth_speed_control(self.smoothness)
    self.right_smooth_speed_controller = CONTROL.smoooth_speed_control(self.smoothness)
    self.left_speed_command, self.right_speed_command = SPEED, SPEED
    self.lat_err = 0.0
    self.head_err = 0.0
    pass


  def process(self, data, decision=0): 
    road_gradient = data.slope
    road_target_point_x = data.target_x
    road_target_point_y = data.target_y
    a = data.poly_2
    b = data.poly_1
    c = data.poly_0

    real_a = (a * ((640/p) ** 2)) * (q / 480)
    real_b = b * (640 / p) * (q / 480)
    real_c = c

    road_target_point = (road_target_point_x, road_target_point_y)
    car_center_point = (car_center_point_x, car_center_point_y)  
   
    target_slope = CONTROL.calculate_slope_between_points(road_target_point, car_center_point)


    #Case 3: 범퍼가 아예 보이지 않는 경우 (범퍼가 보이지는 않지만, 범퍼의 이미지상 x좌표는 알아둬야 함)
    ref_x = real_a * ((q + D_3) ** 2) + real_b * (q + D_3) + real_c
    car_x = car_center_point_x * p / 640

    self.lat_err = ref_x - car_x
    self.head_err = -math.atan(2 * real_a * (q + D_3) + real_b) # 2차함수 미분, road_gradient와 부호 맞춰줌

    #Case 2: 범퍼가 조금 보이는 경우 (카메라가 바라보는 방향과 차체 방향은 같다고 가정)
    ref_x = real_a * ((car_center_point_y +D_2) ** 2) + real_b * (car_center_point_y +D_2) + real_c
    car_x = car_center_point_x * p / 640
    self.lat_err = ref_x - car_x
    self.head_err = -math.atan(2 * real_a * (q + D_3) + real_b) # 2차함수 미분, road_gradient와 부호 맞춰줌

    #Case 1: 범퍼와 전륜축이 모두 보이는 경우 (best) (카메라 끝을 전륜축으로 맞춤)
    ref_x = (road_target_point_x + road_target_point_y)/2 # road_target_point가 0번 인덱스라는 가정
    car_x = car_center_point_x * p / 640
    self.lat_err = ref_x - car_x
    self.head_err = road_gradient

    k = 0.1

    delta = self.head_err+ math.degrees(math.atan(k*self.lat_err/(SPEED)))

    ## Start (여기만 수정하세요)
    # steering_command를 -20도부터 10도까지라고 가정
    # 직접 확인해 봐야 될듯 (-7부터 7까지 각도들)
    # -7, -6에 해당하는 각도 사이에 delta가 위치한다면 둘 중 더 가까운 각으로 해야 됨 (차량의 한계)
    steering_command = 0
    
    # road gradient를 이용할 때 예시
    if road_gradient < 0:
      steering_command = -7

    elif road_gradient == 0:
      steering_command = 0

    else:
      steering_command = 7

    ## END 
    
    actual_left_speed_command = self.left_smooth_speed_controller.update_actual_speed(self.left_speed_command)
    actual_right_speed_command = self.right_smooth_speed_controller.update_actual_speed(self.right_speed_command)
    
    print('target point: ', road_target_point)
    print('car center point: ', car_center_point)
    print('target slope: ', target_slope)
    print('steering command: ', steering_command)
    print('speed command: ', actual_left_speed_command)
    
    return steering_command, actual_left_speed_command, actual_right_speed_command
  

class GenerateSignalNode(Node):
  def __init__(self, sub_topic=SUB_TOPIC_NAME, pub_topic=PUB_TOPIC_NAME, timer=TIMER, que=QUE):
    super().__init__('node_signal_generator')
    
    self.declare_parameter('sub_topic', sub_topic)
    self.declare_parameter('pub_topic', pub_topic)
    self.declare_parameter('timer', timer)
    self.declare_parameter('que', que)
    
    self.sub_topic = self.get_parameter('sub_topic').get_parameter_value().string_value
    self.pub_topic = self.get_parameter('pub_topic').get_parameter_value().string_value
    self.timer_period = self.get_parameter('timer').get_parameter_value().double_value
    self.que = self.get_parameter('que').get_parameter_value().integer_value

    self.is_running = False
    self.br = CvBridge()
    self.generate = IntegrateInfo()
    
    qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, durability=QoSDurabilityPolicy.VOLATILE, depth=self.que)
    self.subscription = self.create_subscription(LaneInfo, self.sub_topic, self.data_callback, qos_profile)
    self.publisher_ = self.create_publisher(Motor, self.pub_topic , self.que)
 
  def data_callback(self, msg):
    angle, left, right = self.generate.process(msg)

    motor = Motor()
    
    motor.angle = angle
    motor.left_speed = left
    motor.right_speed = right
    
    self.publisher_.publish(motor)


def main(args=None):
    rclpy.init(args=args)
    node = GenerateSignalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
        pass
    node.destroy_node()
    rclpy.shutdown()
  
if __name__ == '__main__':
    main()
