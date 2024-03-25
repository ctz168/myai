# -*- coding: utf-8 -*-
import json
import math
import os
import platform
import random
import struct

import cv2
import numpy as np
import pyaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def push(self, state, action, reward, next_state, audio_state, next_audio_state, done, memory, attention):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, audio_state, next_audio_state, done, memory, attention))

    def get_exp(self, batch_size):
        experiences = self.buffer[-batch_size:]
        states = torch.cat([exp[0] for exp in experiences], dim=0)
        actions = torch.cat([exp[1].unsqueeze(0) for exp in experiences], dim=0)
        rewards = torch.tensor([exp[2] for exp in experiences])
        next_states = torch.cat([exp[3] for exp in experiences], dim=0)
        audio_states = torch.cat([exp[4] for exp in experiences], dim=0)
        next_audio_states = torch.cat([exp[5] for exp in experiences], dim=0)
        dones = torch.tensor([int(exp[6]) for exp in experiences])
        memorys = torch.cat([exp[7][0] for exp in experiences], dim=0)
        attentions = torch.cat([exp[8][0] for exp in experiences], dim=0)
        # states, actions, rewards, next_states, dones = zip(*experiences)
        return states, actions, rewards, next_states, audio_states, next_audio_states, dones, memorys, attentions

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        states = torch.cat([exp[0] for exp in experiences], dim=0)
        actions = torch.cat([exp[1].unsqueeze(0) for exp in experiences], dim=0)
        rewards = torch.tensor([exp[2] for exp in experiences])
        next_states = torch.cat([exp[3] for exp in experiences], dim=0)
        audio_states = torch.cat([exp[4] for exp in experiences], dim=0)
        next_audio_states = torch.cat([exp[5] for exp in experiences], dim=0)
        dones = torch.tensor([int(exp[6]) for exp in experiences])
        memorys = torch.cat([exp[7][0] for exp in experiences], dim=0)
        attentions = torch.cat([exp[8][0] for exp in experiences], dim=0)
        # states, actions, rewards, next_states, dones = zip(*experiences)
        return states, actions, rewards, next_states, audio_states, next_audio_states, dones, memorys, attentions

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


class RobotEnv():
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
        self.epsilon = 0
        self.epsilon_decay = 0

    def reset(self):
        # 重置环境状态
        self.state, self.audio_state = self.get_state()
        return self.state, self.audio_state

    def get_state(self):
        return get_Video(), get_audio()

    def step(self, action, t, total_timestip):
        # 执行动作并返回新状态、奖励、完成标志和额外信息
        # ... 执行动作的代码 ..

        # 选择动作
        if self.epsilon is not None and random.random() < self.epsilon:
            print('探索：随机选择一个动作')
            action = torch.rand(4, global_outputdim)  # 随机动作与outputdim要一致

        video_data = action[0].unsqueeze(0)
        audio_data = action[1].unsqueeze(0)

        self.epsilon = self.epsilon * self.epsilon_decay
        # threading.Thread(target=play_video, args=(video_data)).start()  # 调用 play_video 函数进行播放
        play_video(video_data)
        # thread = threading.Thread(target=play_audio, args=(audio_data)).start() # 调用 play_audio 函数进行播放
        play_audio(audio_data)

        # 更新 self.memory 以供下一次调用 step 方法时使用

        next_state, next_audio_state = self.get_state()
        done = False  # 假设环境不会结束
        info = {
            'some_key': 'some_value',
            'another_key': 'another_value',
            # ...其他键值对...
        }
        # 额外信息
        # 如果是最后一个时间步，播放询问音频
        if t == total_timestip - 1:
            print("我干得好吗？此处已经注释")
            # text_to_speech("我干得好吗？")
            # 使用识别出的奖励信号,如果没有提供,则使用默认值
            reward = identify_reward(audio_state)
        else:
            reward = 0
        return next_state, next_audio_state, reward, done, info


class ComplexMultiModalNN(nn.Module):
    def __init__(self):
        # 初始化卷积层
        super(ComplexMultiModalNN, self).__init__()
        outputdim = global_outputdim
        # 初始化LSTM层
        # 假设我们使用一个LSTM来维护自注意力隐藏状态
        self.acttention_lstm = nn.LSTMCell(input_v_a_size, input_v_a_size)
        # 初始化记忆更新决策层
        self.memory_cell = nn.LSTMCell(inputfeaturesnum + 4 * outputdim, inputfeaturesnum + 4 * outputdim)
        self.memory = None
        self.attention = None
        self.optimizer = None
        self.actions = torch.rand(4, global_outputdim)  # 随机动作与outputdim要一致
        # 视觉处理部分
        # 添加视频卷积层
        conv_output_channels = 32  # 卷积层的输出通道数

        self.conv1 = nn.Conv2d(3, conv_output_channels, kernel_size=3, stride=1, padding=1)
        num_features = conv_output_channels * raw_video_height // 2 * raw_video_width // 2  # 计算实际的特征数量
        self.visual_fc = nn.Linear(num_features, input_video_size)
        self.audio_fc = nn.Linear(1024, input_audio_size)  # 假设你需要将音频特征从1024维降到128维
        self.combine_fc = nn.Linear(inputfeaturesnum + inputfeaturesnum + 4 * outputdim,
                                    inputfeaturesnum + inputfeaturesnum + 4 * outputdim)
        # 初始化条件自注意力层
        self.conditional_attention = SelfAttention(embed_size=input_v_a_size, heads=4, condition_size=input_v_a_size)
        # 定义新的输出层，每个维度一个输出头
        self.output_heads = nn.ModuleList([
            nn.Linear(inputfeaturesnum + inputfeaturesnum + 4 * outputdim, outputdim),  # 视频数据，假设输出为64维
            nn.Linear(inputfeaturesnum + inputfeaturesnum + 4 * outputdim, outputdim),  # 语音数据，假设输出为1维
            nn.Linear(inputfeaturesnum + inputfeaturesnum + 4 * outputdim, outputdim),  # 思路（动作长线规划）
            nn.Linear(inputfeaturesnum + inputfeaturesnum + 4 * outputdim, outputdim)  # 第一个是训练、第二个是左手臂、第三是右边手臂、双腿和脚移动数据
        ])
        self.update_memory_decision = nn.Linear(inputfeaturesnum + inputfeaturesnum + 4 * outputdim, 1)
        # 定义Q值输出层，输出维度与动作空间的维度相同
        self.q_value_head = nn.Linear(inputfeaturesnum + inputfeaturesnum + 4 * outputdim, 4)  # 假设最终特征维度为512，动作维度为4

    def forward(self, visual_input, audio_input, memory=None, attention=None, actions=None):
        # 计算 Q 值或找出上次动作值
        q_values = 0
        combined_outputs = None
        if torch.cuda.is_available():
            visual_input = visual_input.cuda()
            audio_input = audio_input.cuda()

        # 视觉特征提取
        visual_input = self.conv1(visual_input)
        visual_input = F.relu(visual_input)
        visual_input = F.max_pool2d(visual_input, 2)
        batch_size = visual_input.size(0)
        visual_input_flattened = visual_input.view(batch_size, -1)  # 展平为 [batch_size, num_features]
        visual_features = self.visual_fc(visual_input_flattened)
        # 听觉特征提取
        audio_features = F.relu(audio_input)
        audio_features = self.audio_fc(audio_features.view(batch_size, -1))

        combined_features = torch.cat((visual_features, audio_features), dim=1)

        if self.memory is None:
            self.memory, self.attention = load_model_memory(self.memory, self.attention, env)
        if memory is None:
            memory = self.memory
            attention = self.attention

        # 应用动态自注意力机制和记忆
        if batch_size == 1:
            combined_features = self.conditional_attention(combined_features, combined_features, combined_features,
                                                           None, attention[1].repeat(1, 1))
            combined_input = combined_features.view(batch_size, inputfeaturesnum)
            new_memory_input = torch.cat(
                (combined_input, self.actions.view(1, self.actions.size(0) * self.actions.size(1))), dim=1)
            combined_input = torch.cat((combined_input, memory[0].repeat(1, 1)), dim=1)
            combined_input = self.combine_fc(combined_input)
            combined_input = F.relu(combined_input)
            outputs = [head(combined_input) for head in self.output_heads]
            combined_outputs = torch.cat(outputs, dim=0)
        else:
            combined_features = self.conditional_attention(combined_features, combined_features, combined_features,
                                                           None, attention)
            combined_input = combined_features.view(batch_size, 1280)
            combined_input = torch.cat((combined_input, memory), dim=1)

            combined_input = self.combine_fc(combined_input)
            combined_input = F.relu(combined_input)
            # combined_input=torch.cat((combined_input,actions.view(batch_size,actions.size(1)*actions.size(2))),dim=1)
            q_values = self.q_value_head(combined_input)  # 使用 Q 值头计算 Q 值

        # 更新记忆状态
        if batch_size == 1:
            # 更新自注意力隐藏层
            attention_to_lstm = combined_features[:, 0, :]
            self.attention = self.acttention_lstm(attention_to_lstm, attention)
            # 如果决定更新记忆，或者memory是None（第一次调用时）
            update_memory_decision = torch.sigmoid(self.update_memory_decision(combined_input))
            should_update_memory = torch.any(update_memory_decision > 0.5)
            if should_update_memory:  # 决定是否更新记忆
                self.memory = self.memory_cell(new_memory_input, memory)
        return combined_outputs, q_values
    # 保存模型记忆状态


def play_video(video_data):
    video_data = torch.clamp(video_data, 0, 1)
    video_output_reshaped = video_data.view(int(math.sqrt(global_outputdim // 3)),
                                            int(math.sqrt(global_outputdim // 3)), 3)
    # 将浮点数张量转换为 uint8 类型的 NumPy 数组
    # 假设值已经被缩放到 [0, 1] 的范围内
    video_output_8bit = (video_output_reshaped * 255).byte().cpu().numpy()
    cv2.imshow('GenerateVideo', video_output_8bit)
    cv2.waitKey(10)


def play_audio(audio_data):
    audio_data = torch.clamp(audio_data, -1, 1)  # 首先确保 audio_data 的值在 [-1, 1] 范围内
    audio_data_int16 = (audio_data.float() * np.iinfo(np.int16).max).to(torch.int16)  # 将浮点数张量转换为 int16 类型
    audio_data_int16_np = audio_data_int16.squeeze(0).numpy()  # 我们需要将其转换为 NumPy 数组并去除批次维度 (1,) 以便使用 pyaudio 播放

    # 初始化 PyAudio 实例
    p = pyaudio.PyAudio()
    # 音频流参数
    stream_params = {
        'format': pyaudio.paInt16,
        'channels': 1,  # 单声道
        'rate': 44100  # 假设的采样率
    }
    stream = p.open(**stream_params, output=True)  # 打开一个流（stream）以播放音频
    stream.write(audio_data_int16_np.tobytes())  # 播放音频
    # 关闭 PyAudio 流
    stream.stop_stream()
    stream.close()
    p.terminate()


# def text_to_speech(text, language='Chinese'):
#     # 初始化语音引擎
#     engine = pyttsx3.init()
#     # 设置语言
#     engine.setProperty('voice', language)
#     # 说一句话
#     engine.say(text)
#     # 运行语音引擎
#     engine.runAndWait()
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
    video_frame = cv2.resize(video_frame, (raw_video_width, raw_video_height))
    if showCamera:
        cv2.imshow('Frame', cv2.resize(video_frame, (raw_video_width, raw_video_height)))
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
    try:
        audio_data = audio_stream.read(512)  # 确保audio_stream是一个已经打开的音频流
    except OSError as e:
        if e.errno == -9981:
            print("音频输入溢出，重置音频流...")  # 关闭当前音频流
        # 创建新的音频流
        audio_stream = pyaudio.PyAudio().open(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100, input=True,
            frames_per_buffer=512)
        # 保持原来的设置
        audio_data = audio_stream.read(512)
    audio_stream.stop_stream()
    audio_stream.close()  # 终止PyAudio实例
    alen = len(audio_data)
    # 检查读取的音频数据长度
    if alen < 1024:
        # 如果读取的数据不足1024字节，填充剩余的部分
        audio_data += b'\x00' * (1024 - len(audio_data))
    # 检查读取的音频数据长度
    if alen > 1024:
        # 如果读取的数据超过1024字节，截断或分帧处理
        # 这里我们选择截断数据
        audio_data = audio_data[:1024]
    # 使用struct.unpack处理音频数据
    audio_data = struct.unpack('b' * 1024, audio_data)
    audio_data = np.array(audio_data) / 32768.0
    audio_data = torch.tensor(audio_data, dtype=torch.float)
    audio_data = audio_data.clone().detach().float()
    return audio_data


def load_model_memory(memory, attention, env):
    if os.path.exists(memory_filename):
        with open(memory_filename, 'r') as f:
            memory_states = json.load(f)
            # 假设隐藏状态和细胞状态是两个张量
            cell_state = torch.tensor(memory_states['cell'])
            hidden_state = torch.tensor(memory_states['hidden'])
            env.epsilon = memory_states['epsilon']
            env.epsilon_decay = memory_states['epsilon_decay']
        memory = (cell_state, hidden_state)
        print(f'Model memory loaded from {memory_filename}、{attention_memory_filename}')
    else:
        memory = (torch.zeros(1, inputfeaturesnum + 4 * global_outputdim),
                  torch.zeros(1, inputfeaturesnum + 4 * global_outputdim))

    if os.path.exists(attention_memory_filename):
        with open(attention_memory_filename, 'r') as f:
            attention_memory = json.load(f)
            attention_cell = torch.tensor(attention_memory['cell'])
            attention_hidden = torch.tensor(attention_memory['hidden'])
        attention = (attention_cell, attention_hidden)
        print(f'Model memory loaded from {attention_memory_filename}')
    else:
        print(f'No model memory found at {memory_filename}、{attention_memory_filename}. Creating new memory.')
        attention = (torch.zeros(1, input_v_a_size), torch.zeros(1, input_v_a_size))
    return memory, attention


def save_model_memory(memory, attention, env):
    memory_states = {'cell': memory[0].tolist(), 'hidden': memory[1].tolist(), 'epsilon': env.epsilon,
                     'epsilon_decay': env.epsilon_decay}
    with open(memory_filename, 'w') as f:
        json.dump(memory_states, f)

    attention_memory = {'cell': attention[0].tolist(), 'hidden': attention[1].tolist()}
    with open(attention_memory_filename, 'w') as f:
        json.dump(attention_memory, f)

    print(f'Model memory saved to {memory_filename}，{attention_memory_filename}')


def action_within_timestep(model, env, total_timestip, replay_buffer, state, audio_state, total_reward):
    buffer_count = 0
    done = False
    for t in range(total_timestip):  # 假设每个episode有1000个时间步

        print("时间步：", t)
        # 选择动作
        model.eval()
        with torch.no_grad():
            video_tensor = get_Video()
            audio_tensor = get_audio()
            action, _ = model(video_tensor, audio_tensor)
            model.actions = action

        # 执行动作
        result = env.step(action, t, total_timestip)
        if result is not None:
            next_state, next_audio_state, reward, done, info = result

            replay_buffer.push(state, action, reward, next_state, audio_state, next_audio_state, done, model.memory,
                               model.attention)
            state = next_state
            audio_state = next_audio_state
            buffer_count = buffer_count + 1

        # 如果episode结束，跳出循环
        if done:
            break
    return buffer_count


def train_model(model, buffer_count, replay_buffer, total_train_num):
    # 训练模型
    print("实施训练")
    model.train()
    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 均方误差损失

    states, actions, rewards, next_states, audio_states, next_audio_states, dones, memorys, attentions = replay_buffer.get_exp(
        buffer_count)

    q_values_cur = model(states, audio_states, memorys, attentions, actions)[1]
    q_values_next = model(next_states, next_audio_states, memorys, attentions, actions)[1]  # 获取下一个状
    # 折扣因子
    gamma = 0.99
    # 计算目标权重
    # 目标权重是立即奖励加上折扣后的未来最大预期Q值
    target_q_values = rewards.unsqueeze(1).repeat(1, q_values_next.size(1)) + (
            1.0 - dones.unsqueeze(1).repeat(1, q_values_next.size(1))) * gamma * q_values_next
    # 现在 target_weights 包含了每个样本的目标权重
    model.optimizer.zero_grad()  # 清空之前的梯度
    loss = criterion(q_values_cur, target_q_values)  # 计算损失
    loss.backward()  # 反向传播
    model.optimizer.step()  # 更新权重
    print(f'Epoch [{buffer_count}], Loss: {loss.item():.4f}')

    if total_train_num % 10 == 0:
        torch.save(model.state_dict(), model_path)
        save_model_memory(model.memory, model.attention, env)


total_timestip = 50
raw_video_height = 224
raw_video_width = 224
input_video_size = 512
input_audio_size = 128
input_v_a_size = input_video_size + input_audio_size
inputfeaturesnum = 1280
total_train_num = 1
epsilon = 0.001  # 初始探索率
epsilon_decay = 0.99  # 探索率衰减因子
showCamera = False
# JSON文件名用于存储模型记忆状态
memory_filename = 'model_memory.json'
attention_memory_filename = 'attention_model_memory.json'
model_path = 'robot_model.pt'
global_outputdim = 20 * 20 * 3
# 初始化摄像头
if platform.system() == "Darwin":  # 判断操作系out统是否为macOS
    cap = cv2.VideoCapture(0)
else:
    webcamipport = 'http://192.168.1.110:8080/video'
    cap = cv2.VideoCapture(webcamipport)
if showCamera:
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Frame', raw_video_width, raw_video_height)

cv2.namedWindow('GenerateVideo', cv2.WINDOW_NORMAL)
cv2.resizeWindow('GenerateVideo', 100, 100)
cv2.waitKey(100)

# 初始化模型
model = ComplexMultiModalNN()
# print(model)
model.optimizer = optim.Adam(model.parameters(), lr=0.001)
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path), strict=False)
    print('Loaded model state from {model_path}')
else:
    print('No model state found at {model_path}. Starting training from scratch.')
# 初始化环境
env = RobotEnv()
# 训练循环
while True:
    # 重置环境和记忆
    replay_buffer = ReplayBuffer(capacity=1000000)
    state, audio_state = env.reset()
    total_reward = 0
    print("开始循环训练：", total_train_num)
    # 执行动作并收集经验
    buffer_count = action_within_timestep(model, env, total_timestip, replay_buffer, state, audio_state, total_reward)
    # threading.Thread(target=train_model, args=(model,buffer_count,replay_buffer,total_train_num)).start()
    train_model(model, buffer_count, replay_buffer, total_train_num)
    # 输出信息
    print("完成一次训练")
    total_train_num = total_train_num + 1
