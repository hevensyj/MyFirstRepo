import cv2
import rclpy
import numpy as np
from math import *
from collections import deque
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge 
from interfaces_pkg.msg import LaneInfo
from .lib import extract_road_info as RF
from .lib import filter_image as FILTER

#---TODO-------------------------------------
SUB_TOPIC_NAME = "topic_masking_img"
PUB_TOPIC_NAME = "topic_lane_info"

TIMER = 0.1
QUE = 1
#--------------------------------------------
class PostProcess():
  def __init__(self):
    pass

  def process(self, img):
    # 버드아이뷰
    (w, h) = (img.shape[1], img.shape[0])  # 640 x 480
    x = 40
    y = 120
    # src_mat : before 좌상,좌하,우상,우하
    src_mat = [[x, h - y],[0 , h], [w - x, h - y], [w, h]]
    # dst_mat : after
    dst_mat = [[0,0],[0,h],[w,0],[w,h]] 
    bird_img = FILTER.bird_convert(img, srcmat=src_mat, dstmat=dst_mat)
    ret,binart_image = cv2.threshold(bird_img, 0, 255, cv2.THRESH_BINARY)
    # 가우시안 블러 (노이즈 좀 처리해주는거)
    img_blur = cv2.GaussianBlur(binart_image, (3,3),2.38)
    # gray 스케일 적용
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    # canny_edgy로 직선검출 
    img_edge = cv2.Canny(img_gray, 10, 30)
    
    # # 커널 정의 (선의 두께 조절)
    # kernel_size = 5  # 커널 크기 (홀수로 설정)
    # kernel = np.ones((kernel_size, kernel_size), np.uint8)  # 커널 정의
    # # 에지 팽창
    # img_dilated = cv2.dilate(img_edge, kernel, iterations=3)  # iterations를 늘리면 선의 두께가 두꺼워집니다
    
    # roi로 관심영역 설정
    # roi_img = FILTER.roi_rectangle_below(img_edge, 300)  #기존 300
    # result_img = roi_img
    result_img = img_edge

    # 디버깅 하는 곳
    # cv2.imshow("bird", bird_img)
    # cv2.imshow("blur",img_blur)
    # cv2.imshow("gray",img_gray)
    # cv2.imshow("canny",img_edge)
    cv2.imshow('postprocess_image', result_img)

    cv2.waitKey(1)
    return result_img
# return result_img

class ExtractInfo():
  def __init__(self):
    self.w, self.h = 640, 480
    self.hough_threshold, self.min_length, self.min_gap = 5, 5, 5 # 기존:10,20,10
    self.angle_tolerance = np.radians(15)
    self.h_mid = int(self.h/2)
    self.cluster_threshold = 20

    self.angle = 0.0
    self.prev_angle = deque([0.0], maxlen=5)
    self.lane = np.array([40, 320, 600])
    self.prev_lanes=[self.lane]    

    self.dist = 220

    self.red, self.green, self.blue = (0, 0, 255), (0, 255, 0), (255, 0, 0)

  def set_height(self, height):
      self.h_mid = height

  def hough(self, img):
      lines = cv2.HoughLinesP(img, 1, np.pi/180, self.hough_threshold, self.min_gap, self.min_length)
      hough_img = np.zeros((img.shape[0], img.shape[1], 3))
      if lines is not None:
          for x1, y1, x2, y2 in lines[:, 0]:
              cv2.line(hough_img, (x1, y1), (x2, y2), self.red, 2)
      cv2.imshow("hough", hough_img)        
      return lines

  def filter(self, lines, img):
      thetas, positions = [], []

      filter_img = np.zeros((img.shape[0], img.shape[1], 3))
      if lines is not None:
          for x1, y1, x2, y2 in lines[:, 0]:
              if y1 == y2:
                  continue
              flag = 1 if y1-y2 > 0 else -1
              theta = np.arctan2(flag * (x2-x1), 0.9*flag * (y1-y2))  #키우면 작은 변화에 둔감: 작은 기울기 변화나 노이즈에 덜 민감해집니다.더 강한 선 검출: 환경 노이즈나 작은 변화에 영향을 덜 받으므로, 더 강한 선분만 검출됩니다.
              if abs(theta - self.angle) < self.angle_tolerance:
                  position = float((x2-x1)*(self.h_mid-y1))/(y2-y1) + x1
                  thetas.append(theta)
                  positions.append(position) 
                  cv2.line(filter_img, (x1, y1), (x2, y2), self.red, 2)
      self.prev_angle.append(self.angle)
      if thetas:
          self.angle = np.mean(thetas)
      #cv2.imshow("filter", filter_img)      
      return positions

  def get_cluster(self, positions):
      clusters = []
      for position in positions:
          if -30 <= position < 260:
              for cluster in clusters:
                  if abs(np.median(cluster) - position) < self.cluster_threshold:
                      cluster.append(position)
                      break
              else:
                  clusters.append([position])
      lane_candidates = [np.mean(cluster) for cluster in clusters]
      # print('lane_candidates', lane_candidates)
      return lane_candidates

  def predict_lane(self):
      predicted_lane = self.lane[1] + [-self.dist/max(np.cos(self.angle), 0.75), 0, self.dist/max(np.cos(self.angle), 0.75)]
      predicted_lane = predicted_lane + (self.angle - np.mean(self.prev_angle))*70
      # print('predicted_lane', predicted_lane)
      return predicted_lane

  def update_lane(self, lane_candidates, predicted_lane):
    if not lane_candidates:
        self.lane = predicted_lane
        return
    possibles = []
    for lc in lane_candidates:
        idx = np.argmin(abs(self.lane-lc))
        if idx == 0:
            estimated_lane = [lc, lc + self.dist/max(np.cos(self.angle), 0.75), lc + (2*self.dist)/max(np.cos(self.angle), 0.75)]
            lc2_candidate, lc3_candidate = [], []
            for lc2 in lane_candidates:
                if abs(lc2-estimated_lane[1]) < 50 :
                    lc2_candidate.append(lc2)
            for lc3 in lane_candidates:
                if abs(lc3-estimated_lane[2]) < 50 :
                    lc3_candidate.append(lc3)
            if not lc2_candidate:
                lc2_candidate.append(estimated_lane[1])
            if not lc3_candidate:
                lc3_candidate.append(estimated_lane[2])
            for lc2 in lc2_candidate:
                for lc3 in lc3_candidate:
                    possibles.append([lc, lc2, lc3])
    
        elif idx == 1:
            estimated_lane = [lc - self.dist/max(np.cos(self.angle), 0.75), lc, lc + self.dist/max(np.cos(self.angle), 0.75)]
            lc1_candidate, lc3_candidate = [], []
            for lc1 in lane_candidates:
                if abs(lc1-estimated_lane[0]) < 50 :
                    lc1_candidate.append(lc1)
            for lc3 in lane_candidates:
                if abs(lc3-estimated_lane[2]) < 50 :
                    lc3_candidate.append(lc3)
            if not lc1_candidate:
                lc1_candidate.append(estimated_lane[0])
            if not lc3_candidate:
                lc3_candidate.append(estimated_lane[2])
            for lc1 in lc1_candidate:
                for lc3 in lc3_candidate:
                    possibles.append([lc1, lc, lc3])
    
        else :
            estimated_lane = [lc - (2*self.dist)/max(np.cos(self.angle), 0.75), lc - self.dist/max(np.cos(self.angle), 0.75), lc]
            lc1_candidate, lc2_candidate = [], []
            for lc1 in lane_candidates:
                if abs(lc1-estimated_lane[0]) < 50 :
                    lc1_candidate.append(lc1)
            for lc2 in lane_candidates:
                if abs(lc2-estimated_lane[1]) < 50 :
                    lc2_candidate.append(lc2)
            if not lc1_candidate:
                lc1_candidate.append(estimated_lane[0])
            if not lc2_candidate:
                lc2_candidate.append(estimated_lane[1])
            for lc1 in lc1_candidate:
                for lc2 in lc2_candidate:
                    possibles.append([lc1, lc2, lc])
    
    possibles = np.array(possibles)
    # print('possibles', possibles)
    error = np.sum((possibles-predicted_lane)**2, axis=1)
    best = possibles[np.argmin(error)]
    self.lane = 0.4 * best + 0.6 * predicted_lane
    self.mid = np.mean(self.lane)

  def mark_lane(self, img, lane=None):
      
      img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

      if lane is None:
          lane = self.lane
          self.mid = self.lane[1]
      l1, l2, l3 = self.lane


      cv2.circle(img_rgb, (int(l1), self.h_mid), 3, self.red, 5, cv2.FILLED)
      cv2.circle(img_rgb, (int(l2), self.h_mid), 3, self.green, 5, cv2.FILLED)
      cv2.circle(img_rgb, (int(l3), self.h_mid), 3, self.blue, 5, cv2.FILLED)

      cv2.imshow('marked', img_rgb)
  
  def process(self, img):
    """
    lines = self.hough(img)
    positions = self.filter(lines, img)
    lane_candidates = self.get_cluster(positions)
    predicted_lane = self.predict_lane()
    self.update_lane(lane_candidates, predicted_lane)
    self.mark_lane(img)
    cv2.waitKey(1)

    grad = np.float32(self.angle)
    left_x = int(self.lane[0])
    middle_x = int(self.lane[1])
    right_x = int(self.lane[2])
    """

    # 수정 코드 시작
    grad_list = []
    left_x_list = []
    right_x_list = []
    middle_x_list = []

    road_x = []
    road_y = []
    for h in range(481):  # h_mid를 0부터 480까지 변화시키면서 road_x, road_y 생성
        self.set_height(h)
        lines = self.hough(img)  # img는 이미지 데이터가 전달되어야 합니다.
        positions = self.filter(lines, img)
        lane_candidates = self.get_cluster(positions)
        predicted_lane = self.predict_lane()
        self.update_lane(lane_candidates, predicted_lane)
        grad = np.float32(self.angle)
        left_x = int(self.lane[0])
        middle_x = int(self.lane[1])
        right_x = int(self.lane[2])

        grad_list.append(grad)
        left_x_list.append(left_x)
        right_x_list.append(right_x)
        middle_x_list.append(middle_x)

        road_x.append((left_x + right_x) / 2)
        road_y.append(h)
        self.mark_lane(img)
        cv2.waitKey(1)

    coefficients = np.polyfit(road_y, road_x, 2)
    a = coefficients[0]
    b = coefficients[1]
    c = coefficients[2]

    return grad_list[0], left_x[0], middle_x[0], right_x[0], a, b, c
    #수정 코드 끝
    
class ExtractInfoNode(Node):
  def __init__(self, sub_topic=SUB_TOPIC_NAME, pub_topic=PUB_TOPIC_NAME, timer=TIMER, que=QUE):
    super().__init__('node_info_extraction')
    
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
    self.post = PostProcess()
    self.detect = ExtractInfo()
    
    image_qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, durability=QoSDurabilityPolicy.VOLATILE, depth=self.que)
    self.subscription = self.create_subscription(Image, self.sub_topic, self.image_callback, image_qos_profile)

    self.publisher_ = self.create_publisher(LaneInfo, self.pub_topic , self.que)
    self.timer = self.create_timer(self.timer_period, self.timer_callback)

  def image_callback(self, data):
    self.is_running = True
    current_frame = self.br.imgmsg_to_cv2(data)   # mask image 
    processed_img = self.post.process(current_frame)
    cv2.imshow('yolo',current_frame)

    grad, left_x, middle_x, right_x, a, b, c = self.detect.process(processed_img)
    
    lane = LaneInfo()
    lane.slope = float(grad)
    lane.target_x = round(left_x)
    lane.target_y = round(right_x)

    # LaneInfo.msg 파일에 다항식 정보 추가

    # float32 poly_2
    # float32 poly_1
    # float32 poly_0
    
    lane.poly_2 = a
    lane.poly_1 = b
    lane.poly_0 = c
    # 수정 코드 끝
    
    self.publisher_.publish(lane)

  def timer_callback(self):
    if not self.is_running:
      self.get_logger().info('Not published yet: "%s"' % self.sub_topic)
      
def main(args=None):
    rclpy.init(args=args)
    node = ExtractInfoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
        pass
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()
  
if __name__ == '__main__':
    main()