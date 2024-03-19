# -*- coding: utf-8 -*-
import io
import math
import random
import struct
import json
import os
import cv2
import gym
from gym import spaces
import numpy as np
import pyaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import platform
import threading

# 定义权重学习模型
class WeightLearningModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(WeightLearningModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, action_results):
        x = torch.relu(self.fc1(action_results))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def push(self, state, action, reward, next_state, audio_state, next_audio_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, audio_state, next_audio_state, done))

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        states = torch.cat([exp[0] for exp in experiences], dim=0)
        actions = torch.cat([exp[1].unsqueeze(0) for exp in experiences], dim=0)
        rewards = torch.tensor([exp[2] for exp in experiences])
        next_states = torch.cat([exp[3] for exp in experiences], dim=0)
        audio_states = torch.cat([exp[4] for exp in experiences], dim=0)
        next_audio_states = torch.cat([exp[5] for exp in experiences], dim=0)
        dones = torch.tensor([int(exp[6]) for exp in experiences])


        # states, actions, rewards, next_states, dones = zip(*experiences)
        return states, actions, rewards, next_states, audio_states, next_audio_states, dones

    def __len__(self):
        return len(self.buffer)

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, condition_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.condition_size = condition_size

        # 添加一个线性层来处理条件信息
        self.condition_linear = nn.Linear(condition_size, embed_size)

        # 原有的自注意力层的权重
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask, condition):
        # 处理条件信息
        condition_features = self.condition_linear(condition)
        combined_queries = torch.cat([queries, condition_features], dim=-1)
        N = queries.size(0)
        # 原有的自注意力计算
        values = values.view(N, -1, self.heads, self.head_dim)
        keys = keys.view(N, -1, self.heads, self.head_dim)
        combined_queries = combined_queries.view(N, -1, self.heads, self.head_dim)

        # 计算注意力权重
        attention = torch.matmul(combined_queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(attention, dim=-1)
        out = torch.matmul(attention, values)

        # 将输出展平并经过一个最终的线性层
        out = out.view(N, -1, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out


class RobotEnv(gym.Env):
    def __init__(self):
        super(RobotEnv, self).__init__()
        # 初始化扬声器
        self.speaker = pyaudio.PyAudio().open(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100,
            input=False,
            output=True,
            frames_per_buffer=1024
        )
        # 初始化环境状态
        self.state = None
        self.audio_state = None
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=float)  # 假设视频数据


    def reset(self):
        # 重置环境状态
        self.state, self.audio_state = self.get_initial_state()
        return self.state, self.audio_state

    def get_initial_state(self):
        return get_Video(), get_audio()

    def step(self, action):
        # 执行动作并返回新状态、奖励、完成标志和额外信息
        # ... 执行动作的代码 ..

        # 选择动作
        if epsilon is not None and random.random() < epsilon:
            print('探索：随机选择一个动作')
            action =torch.rand(10, global_outputdim) # 随机动作与outputdim要一致
            video_data = action[0].unsqueeze(0)
            audio_data = action[1].unsqueeze(0)
        else:
            # 使用模型执行动作并获取动作概率和记忆状态
            video_data=action[0]
            audio_data = action[1]
        #threading.Thread(target=play_video, args=(video_data)).start()  # 调用 play_video 函数进行播放
        play_video(video_data)
        thread = threading.Thread(target=play_audio, args=(audio_data)).start() # 调用 play_audio 函数进行播放
        # play_audio(audio_data)

        # 更新 self.memory 以供下一次调用 step 方法时使用

        next_state, next_audio_state = self.get_initial_state()
        done = False  # 假设环境不会结束
        info = {
            'some_key': 'some_value',
            'another_key': 'another_value',
            # ...其他键值对...
        }
        # 额外信息

        # 使用识别出的奖励信号,如果没有提供,则使用默认值
        reward = identify_reward(audio_state)
        return next_state, next_audio_state, reward, done, info


class ComplexMultiModalNN(nn.Module):
    def __init__(self):
        # 初始化卷积层
        super(ComplexMultiModalNN, self).__init__()
        # 初始化LSTM层
        self.memory_cell =nn.LSTMCell(input_value_size, input_value_size)
        # 假设我们使用一个LSTM来维护自注意力隐藏状态
        self.acttention_lstm=nn.LSTMCell(input_value_size, input_value_size)
        # 初始化记忆更新决策层
        self.update_memory_decision = nn.Linear(input_value_size, 1)
        self.memory=None
        self.attention=None
        # 视觉处理部分
        # 添加视频卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 初始化全连接层，根据实际输出形状计算输入特征数量
        output_height = 56  # 实际卷积层输出的高度
        output_width = 56  # 实际卷积层输出的宽度
        conv_output_channels = 64  # 卷积层的输出通道数
        num_features = conv_output_channels * output_height * output_width  # 计算实际的特征数量

        # 定义线性层，使用实际的特征数量
        self.visual_fc = nn.Linear(num_features, input_video_size)
        # 听觉处理部分

        # 添加音频卷积层
        self.audio_conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.audio_conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        # 全连接层
        self.audio_fc = nn.Linear(64 * 1024, input_audio_size)  # 假设你需要将音频特征从1024维降到128维

        # 初始化条件自注意力层
        self.conditional_attention = SelfAttention(embed_size=input_value_size, heads=4, condition_size=input_value_size)

        inputfeaturesnum=1920
        outputdim=global_outputdim
        # 定义新的输出层，每个维度一个输出头
        self.output_heads = nn.ModuleList([
            nn.Linear(inputfeaturesnum, outputdim),  # 视频数据，假设输出为64维
            nn.Linear(inputfeaturesnum, outputdim),  # 语音数据，假设输出为1维
            nn.Linear(inputfeaturesnum, outputdim),  # 双腿和脚移动数据
            nn.Linear(inputfeaturesnum, outputdim),  # 双手臂和手掌移动数据
            nn.Linear(inputfeaturesnum, outputdim),  # 腰部转动数据
            nn.Linear(inputfeaturesnum, outputdim),  # 头部转动数据
            nn.Linear(inputfeaturesnum, outputdim),  # 语音对应的文本数据
            nn.Linear(inputfeaturesnum, outputdim),  # 面部表情数据
            nn.Linear(inputfeaturesnum, outputdim),  # 接入互联网通过TCP的发包数据
            nn.Linear(inputfeaturesnum, outputdim)   # Python代码数据
        ])
        # 定义Q值输出层，输出维度与动作空间的维度相同
        self.q_value_head = nn.Linear(inputfeaturesnum+10*outputdim, 10)  # 假设最终特征维度为512，动作空间维度为10

    def forward(self, visual_input, audio_input, memory=None, attention=None,actions=None):

        if torch.cuda.is_available():
            visual_input = visual_input.cuda()
            audio_input = audio_input.cuda()

        # 视觉特征提取
        # print(" visual_input shape:", visual_input.size())
        visual_input = F.relu(self.conv1(visual_input))
        visual_input = F.max_pool2d(visual_input, 2)
        visual_input = F.relu(self.conv2(visual_input))
        visual_input = F.max_pool2d(visual_input, 2)
        # 在forward方法中，卷积层之后添加打印语句
        # print("Convolutional visual_input shape:", visual_input.size())
        # 计算卷积层输出的特征图的展平大小
        batch_size = visual_input.size(0)
        # print("batch_size:", batch_size)
        # 展平特征图
        visual_input_flattened = visual_input.view(batch_size, -1)  # 展平为 [batch_size, num_features]
        # print("visual_input_flattened:", visual_input_flattened)
        # 全连接层处理展平后的特征图
        visual_features = self.visual_fc(visual_input_flattened)
        # visual_features = visual_features.unsqueeze(1)
        # 听觉特征提取
        audio_features = F.relu(self.audio_conv1(audio_input))
        audio_features = F.relu(self.audio_conv2(audio_features))
        # 此处应包含展平操作，假设音频特征在展平前的最后一维为1024
        # 展平音频特征，确保批次大小为1在最前面
        audio_features = audio_features.permute(0, 2, 1).contiguous().view(batch_size, -1)
        audio_features = self.audio_fc(audio_features)
        # audio_features = audio_features.unsqueeze(1)
        combined_features = torch.cat((visual_features, audio_features), dim=1)
        if self.memory is None:
            self.memory, self.attention = load_model_memory(self.memory, self.attention)

        current_batch_size = combined_features.size(0)
        # 应用动态自注意力机制
        combined_features = self.conditional_attention(combined_features, combined_features, combined_features, None,
                                                       self.attention[1].repeat(current_batch_size, 1))
        combined_input=combined_features.view(current_batch_size, 1280)

        combined_input = torch.cat((combined_input, self.memory[0].repeat(current_batch_size, 1)), dim=1)
        print("combined_input",combined_input.size())
        # 计算动作概率
        outputs = [head(combined_input) for head in self.output_heads]
        combined_outputs = torch.cat(outputs, dim=0)
        # 计算 Q 值
        q_values =0
        if actions is not None:
            combined_input=torch.cat((combined_input,actions.view(current_batch_size,actions.size(1)*actions.size(2))),dim=1)
            q_values = self.q_value_head(combined_input)  # 使用 Q 值头计算 Q 值

        # 更新记忆状态
        if current_batch_size == 1:
            # 更新自注意力隐藏层
            input_to_lstm = combined_features[:, 0, :]
            attention_hidden, attention_cell = self.acttention_lstm(input_to_lstm, attention)
            # 如果决定更新记忆，或者memory是None（第一次调用时）
            update_memory_decision = torch.sigmoid(self.update_memory_decision(combined_features))
            should_update_memory = torch.any(update_memory_decision > 0.5)
            if should_update_memory:
                memory = self.memory_cell(input_to_lstm, memory)
                # 决定是否更新记忆

                # 使用 torch.any() 来检查是否有任何元素大于 0.5

        return combined_outputs, q_values, memory, attention
    # 保存模型记忆状态


def play_video(video_data):
    video_data = torch.clamp(video_data, 0, 1)
    video_output_reshaped = video_data.view(int(math.sqrt( global_outputdim//3)),int(math.sqrt( global_outputdim//3)), 3)
    # 将浮点数张量转换为 uint8 类型的 NumPy 数组
    # 假设值已经被缩放到 [0, 1] 的范围内
    video_output_8bit = (video_output_reshaped * 255).byte().cpu().numpy()

    # 如果需要，将 RGB 转换为 BGR
    frame_to_show_bgr = cv2.cvtColor(video_output_8bit, cv2.COLOR_RGB2BGR)
    cv2.imshow('GenerateVideo',frame_to_show_bgr)

    cv2.waitKey(10)
def play_audio(audio_data):

    audio_data = torch.clamp(audio_data, -1, 1)  # 首先确保 audio_data 的值在 [-1, 1] 范围内
    audio_data_int16 = (audio_data.float() * np.iinfo(np.int16).max).to(torch.int16) # 将浮点数张量转换为 int16 类型
    audio_data_int16_np = audio_data_int16.squeeze(0).numpy() # 我们需要将其转换为 NumPy 数组并去除批次维度 (1,) 以便使用 pyaudio 播放

    # 初始化 PyAudio 实例
    p = pyaudio.PyAudio()
    # 音频流参数
    stream_params = {
        'format': pyaudio.paInt16,
        'channels': 1,  # 单声道
        'rate': 44100  # 假设的采样率
    }
    stream = p.open(**stream_params, output=True)    # 打开一个流（stream）以播放音频
    stream.write(audio_data_int16_np.tobytes())# 播放音频
    # 关闭 PyAudio 流
    stream.stop_stream()
    stream.close()
    p.terminate()
def text_to_speech(text, language='Chinese'):
    # 初始化语音引擎
    engine = pyttsx3.init()
    # 设置语言
    engine.setProperty('voice', language)
    # 说一句话
    engine.say(text)
    # 运行语音引擎
    engine.runAndWait()
def compute_reward(self, next_state):
    # 假设我们有一个目标物体,我们想要机器人将其移动到特定位置
    # next_state包含了目标物体的当前位置
    # 这里我们简单地使用物体位置的变化来计算奖励
    # 假设next_state是一个包含物体位置的数组,例如 [x, y, z]
    object_position = next_state[-3:]  # 假设物体位置是状态的最后三个值
    target_position = np.array([0.0, 0.0, 0.0])  # 目标位置

    # 计算物体位置与目标位置之间的距离
    distance_to_target = np.sqrt(np.sum((object_position - target_position) ** 2))

    # 奖励是距离的倒数,距离越小,奖励越高
    reward = 1.0 / (distance_to_target + 1e-3)  # 加入一个小的常数以避免除以0

    return reward
def check_done(self, next_state):
    # 检查任务是否完成
    # 这里我们简单地检查物体是否到达了目标位置
    object_position = next_state[-3:]  # 同上
    target_position = np.array([0.0, 0.0, 0.0])  # 目标位置

    # 如果物体位置与目标位置足够接近,认为任务完成
    distance_threshold = 0.1  # 定义一个阈值,例如10cm
    if np.all(np.abs(object_position - target_position) < distance_threshold):
        return True

    # 如果没有达到阈值,任务未完成
    return False
def identify_reward(Rewardaudio_data):
    # 未实现对动作的负反馈，比如遇到阻力
    text = ""
    # 解析文本以识别表扬
    if "well done" in text.lower() or "good job" in text.lower():
        return 1.0  # 正面奖励
    elif "not good" in text.lower() or "wrong" in text.lower():
        return -1.0  # 负面奖励
    else:
        return 0.0  # 无奖励

def get_Video():

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    ret, frame = cap.read()  # 确保cap是一个已经打开的视频流
    if not ret:
        raise ValueError("无法从摄像头读取数据")

    # 将BGR图像转换为RGB格式并进行归一化
    video_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_frame = video_frame.astype(np.float32) / 255.0

    # 调整大小和归一化
    video_frame = cv2.resize(video_frame, (224, 224))
    if showCamera:
        cv2.imshow('Frame', video_frame)
        cv2.waitKey(10)
    video_frame = np.transpose(video_frame, (2, 0, 1))

    # 确保视频数据的形状是 [1, channels, height, width]
    video_data = torch.tensor(video_frame).unsqueeze(0).float()
    return video_data

def get_audio():
    # 尝试读取音频数据，添加异常处理

    audio_stream = pyaudio.PyAudio().open(
        format=pyaudio.paInt16,
        channels=1,
        rate=44100,
        input=True,
        frames_per_buffer=512  # 保持原来的设置
    )
    audio_data = audio_stream.read(512)  # 确保audio_stream是一个已经打开的音频流
    audio_stream.stop_stream()
    audio_stream.close()  # 终止PyAudio实例
    alen=len(audio_data)
    # 检查读取的音频数据长度
    if alen < 1024:
        # 如果读取的数据不足1024字节，填充剩余的部分
        audio_data += b'\x00' * (1024 - len(audio_data))
    # 检查读取的音频数据长度
    if alen > 1024:
        # 如果读取的数据超过1024字节，截断或分帧处理
        # 这里我们选择截断数据
        audio_data = audio_data[:1024]
    if isinstance(audio_data, torch.Tensor):
        print("audio_data is a PyTorch tensor.")
    else:
        # 使用struct.unpack处理音频数据
        audio_data = struct.unpack('b' * 1024, audio_data)
        audio_data = np.array(audio_data) / 32768.0
        # 将音频数据转换为适合卷积层的形状
        audio_data = np.reshape(audio_data, (1, 1, -1))  # [1, 1, sample_rate]
        audio_data = torch.tensor(audio_data, dtype=torch.float)
    audio_data = audio_data.clone().detach().float()
    return audio_data
def load_model_memory(memory,attention):

        if os.path.exists(memory_filename):
            with open(memory_filename, 'r') as f:
                memory_states = json.load(f)
                # 假设隐藏状态和细胞状态是两个张量
                cell_state = torch.tensor(memory_states['cell'])
                hidden_state = torch.tensor(memory_states['hidden'])
                epsilon = memory_states['epsilon']
                epsilon_decay = memory_states['epsilon_decay']
            memory=(cell_state,hidden_state )
            print(f'Model memory loaded from {memory_filename}、{attention_memory_filename}')
        else:
            memory=(torch.zeros(1, input_value_size), torch.zeros(1, input_value_size))


        if os.path.exists(attention_memory_filename):
            with open(attention_memory_filename, 'r') as f:
                attention_memory = json.load(f)
                attention_cell = torch.tensor(attention_memory['cell'])
                attention_hidden = torch.tensor(attention_memory['hidden'])
            attention=(attention_cell,attention_hidden )
            print(f'Model memory loaded from {attention_memory_filename}')
        else:
            print(f'No model memory found at {memory_filename}、{attention_memory_filename}. Creating new memory.')
            attention = (torch.zeros(1, input_value_size), torch.zeros(1, input_value_size))
        return memory,attention
def save_model_memory(memory,attention):
    memory_states = {'cell': memory[0].tolist(),'hidden': memory[1].tolist(), 'epsilon': epsilon,'epsilon_decay': epsilon_decay}
    with open(memory_filename, 'w') as f:
        json.dump(memory_states, f)

    attention_memory = {'cell': attention[0].tolist(), 'hidden': attention[1].tolist()}
    with open(attention_memory_filename, 'w') as f:
        json.dump(attention_memory, f)

    print(f'Model memory saved to {memory_filename}，{attention_memory_filename}')




input_video_size=512
input_audio_size=128
input_value_size=input_video_size+input_audio_size
# 在训练循环的开始处设置 epsilon
epsilon = 1.0  # 初始探索率
epsilon_decay = 0.99  # 探索率衰减因子
showCamera=True
# JSON文件名用于存储模型记忆状态
memory_filename = 'model_memory.json'
attention_memory_filename = 'model_memory.json'
model_path='robot_model.pt'
global_outputdim=100*100*3
# 初始化摄像头
if platform.system() == "Darwin":# 判断操作系out统是否为macOS
    cap = cv2.VideoCapture(0)
else:
    webcamipport = 'http://192.168.1.116:8080/video'
    cap = cv2.VideoCapture(webcamipport)
if showCamera:
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Frame', 224, 224)

cv2.namedWindow('GenerateVideo', cv2.WINDOW_NORMAL)
cv2.resizeWindow('GenerateVideo', 100, 100)
cv2.waitKey(1000)

# 初始化模型
model = ComplexMultiModalNN()
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path), strict=False)
    print(f'Loaded model state from {model_path}')
else:
    print(f'No model state found at {model_path}. Starting training from scratch.')
# 初始化优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 初始化环境
env = RobotEnv()

# 训练循环
num_episodes = 20  # 设置一个较大的episode数,以便模型可以持续学习
for episode in range(num_episodes):
    # 重置环境和记忆
    replay_buffer = ReplayBuffer(capacity=10000)
    state, audio_state = env.reset()

    total_reward = 0
    print("开始循环训练：", episode)
    # 执行动作并收集经验
    total_timestip = 10
    buffer_count = 0
    done=False
    for t in range(total_timestip):  # 假设每个episode有1000个时间步

        print("时间步：", t)
        # 选择动作
        model.eval()
        with torch.no_grad():
            video_tensor = get_Video()
            audio_tensor = get_audio()
            action, _ ,memory,attention= model(video_tensor, audio_tensor)

        # 执行动作
        result = env.step(action)
        if result is not None:
            next_state, next_audio_state, reward, done, info = result

            replay_buffer.push(state, action, reward, next_state, audio_state, next_audio_state, done)
            state = next_state
            audio_state = next_audio_state
            total_reward += reward
            buffer_count = buffer_count + 1


        # 如果是最后一个时间步，播放询问音频
        if t == total_timestip - 1:
            print("我干得好吗？此处已经注释")
            # text_to_speech("我干得好吗？")

        # 如果episode结束，跳出循环
        if done:
            break

    # 训练模型
    print("实施训练")
    model.train()
    weight_model = WeightLearningModel(global_outputdim, 1)
    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 均方误差损失
    optimizer = optim.Adam(weight_model.parameters(), lr=0.001)

    for epoch in range(buffer_count):  # 每个episode训练buffer_count次
        # 假设我们的抽样批次大小为4
        batch_size = 4
        states, actions, rewards, next_states, audio_states, next_audio_states, dones = replay_buffer.sample(
            batch_size)
        # 假设我们有以下张量：
        # rewards: (batch_size,) 形状的张量，包含每个样本的立即奖励
        # max_next_q_values: (batch_size,) 形状的张量，包含每个样本下一个状态的最大预期Q值
        # gamma: 折扣因子，一个介于0和1之间的值

        # 示例下一个状态的最大预期Q值
        q_values_next = model(next_states, next_audio_states, memory,attention,actions)[1]  # 获取下一个状
        max_q_values_next = torch.max(q_values_next, dim=1)[0]  # 选择每个样本的最大Q值
        # 折扣因子
        gamma = 0.99
        # 计算目标权重
        # 目标权重是立即奖励加上折扣后的未来最大预期Q值
        target_weights = rewards + gamma * max_q_values_next
        # 现在 target_weights 包含了每个样本的目标权重
        optimizer.zero_grad()  # 清空之前的梯度
        outputs = weight_model(actions)  # 前向传播
        loss = criterion(outputs, target_weights)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')


    # 输出信息
    print("完成一次训练，已保存模型")

    # 动态更新模型文件
    torch.save(model.state_dict(), model_path)
    # 训练结束后更新记忆状态
    save_model_memory(memory,attention,memory_filename)
# 在所有episode完成后，执行资源释放
cap.release()  # 释放摄像头资源

if __name__ == '__main__':
    # ...省略初始化代码...
    # ...省略训练循环...
    # 在程序的最后，执行资源释放
    cap.release()
