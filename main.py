# -*- coding: utf-8 -*-
import io
import random
import string
import struct

import cv2
import gtts
import gym
import numpy as np
import pyaudio
import speech_recognition as sr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class AudioGenerator(nn.Module):
    def __init__(self, input_size, audio_sample_rate, audio_length):
        super(AudioGenerator, self).__init__()
        # 假设输入特征向量的维度等于音频长度
        self.fc = nn.Linear(input_size, audio_length)

    def forward(self, x):
        # 将输入特征向量转换为音频波形
        audio_waveform = self.fc(x)

        # 归一化音频波形到 [-1, 1] 的范围
        audio_waveform = torch.tanh(audio_waveform)

        return audio_waveform


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def push(self, state, action, reward, next_state,audio_state,next_audio_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state,audio_state,next_audio_state, done))

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        states = torch.cat([exp[0] for exp in experiences],dim=0)
        actions_int=[exp[1]  for exp in experiences]
        actions = torch.tensor([(1,action) for action in actions_int], dtype=torch.int64)
        rewards_float=[exp[2]  for exp in experiences]
        rewards = torch.tensor([(1,reward) for reward in rewards_float], dtype=torch.float)
        next_states = torch.cat([exp[3] for exp in experiences],dim=0)
        audio_states= torch.cat([exp[4] for exp in experiences],dim=0)
        next_audio_states = torch.cat([exp[5] for exp in experiences],dim=0)
        dones_int=[ int(exp[6]) for exp in experiences]
        dones = torch.tensor([(1,done) for done in dones_int], dtype=torch.float)

        #states, actions, rewards, next_states, dones = zip(*experiences)
        return states, actions, rewards, next_states, audio_states,next_audio_states,dones

    def __len__(self):
        return len(self.buffer)


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask):
        N = queries.size(0)
        # Split the embeddings into self.heads different pieces
        values = values.view(N, -1, self.heads, self.head_dim)
        keys = keys.view(N, -1, self.heads, self.head_dim)
        queries = queries.view(N, -1, self.heads, self.head_dim)

        # Scaled dot-product attention
        attention = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-1e20"))

        # Apply softmax and scale
        attention = torch.softmax(attention, dim=-1)

        # Apply attention
        out = torch.matmul(attention, values)

        # Concatenate heads and pass through a final linear layer
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
        # 初始化记忆状态
        self.memory =None
        # 初始化环境状态
        self.state = None
        self.audio_state=None
        self.action_space = gym.spaces.Discrete(10)  # 假设有10种可能的动作
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=float)  # 假设视频数据

    def play_audio(self, audio_data):
        # 播放音频数据
        self.speaker.write(audio_data)

    def text_to_speech(self, question_audio_text, language='en-US'):
        # 使用gTTS库将文本转换为音频数据

        tts = gtts.gTTS(text=question_audio_text, lang='zh-CN')  # 根据实际情况调整语言参数
        with io.BytesIO() as output:
            tts.write_to_fp(output)
            audio_data = output.getvalue()
        return audio_data

    def reset(self):
        # 重置环境状态
        self.state,self.audio_state = self.get_initial_state()
        return self.state,self.audio_state

    def get_initial_state(self):
        return get_Video(),get_audio()

    def step(self, action):
        # 执行动作并返回新状态、奖励、完成标志和额外信息
        # ... 执行动作的代码 ..

        # 选择动作
        if epsilon is not None and random.random() < epsilon:
            # 探索：随机选择一个动作
            action = env.action_space.sample()
        else:
            # 使用模型执行动作并获取动作概率和记忆状态

            action_probs, memory = model(state, audio_state, self.memory)
            # 更新 self.memory 以供下一次调用 step 方法时使用
            self.memory = memory
            action = torch.argmax(action_probs).item()

            next_state,next_audio_state = self.get_initial_state()

            done = False  # 假设环境不会结束
            info = {
                'some_key': 'some_value',
                'another_key': 'another_value',
                # ...其他键值对...
            }
            # 额外信息

            # 使用识别出的奖励信号,如果没有提供,则使用默认值
            reward = identify_reward(audio_state)

            return next_state, next_audio_state,reward, done, info


class ComplexMultiModalNN(nn.Module):
    def __init__(self):
        # 初始化卷积层
        super(ComplexMultiModalNN, self).__init__()

        # 初始化LSTM层
        self.memory_cell = nn.LSTMCell(256, 256)  # 假设输入和隐藏状态的维度都是256

        # 初始化记忆更新决策层
        self.update_memory_decision = nn.Linear(256, 1)

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
        self.visual_fc = nn.Linear(num_features, 128)
        # 听觉处理部分

        # 添加音频卷积层
        self.audio_conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.audio_conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        # 全连接层
        self.audio_fc = nn.Linear(65536, 128)  # 假设你需要将音频特征从1024维降到128维


        # 自注意力层
        self.self_attention = SelfAttention(embed_size=128, heads=4)

        # 动作生成部分
        self.action_net = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)  # 假设有10种可能的动作
        )


    def forward(self, visual_input, audio_input, memory=None):

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
        num_features = visual_input.size(1) * visual_input.size(2) * visual_input.size(3)
        # print("num_features:", num_features)
        # 展平特征图
        visual_input_flattened = visual_input.view(batch_size, -1)  # 展平为 [batch_size, num_features]
        # print("visual_input_flattened:", visual_input_flattened)
        # 全连接层处理展平后的特征图
        visual_features = self.visual_fc(visual_input_flattened)

        # 听觉特征提取
        audio_features = F.relu(self.audio_conv1(audio_input))
        audio_features = F.relu(self.audio_conv2(audio_features))
        # print("Audio features shape before flattening:", audio_features.size())
        # 此处应包含展平操作，假设音频特征在展平前的最后一维为1024
        # 展平音频特征，确保批次大小为1在最前面
        audio_features_flattened = audio_features.permute(0, 2, 1).contiguous().view(batch_size, -1)
        # print("Audio features shape after flattening:",audio_features_flattened.size())  # 应该输出 torch.Size([1, 1024])   print("Audio features shape after flattening:", audio_features_flattened.size())
        # assert audio_features_flattened.shape[1] == self.audio_fc.in_features
        audio_features_processed = self.audio_fc(audio_features_flattened)
        audio_features_processed = audio_features_processed.unsqueeze(1)
        # 自注意力
        visual_features = self.self_attention(visual_features, visual_features, visual_features, None)
        audio_features_processed= self.self_attention(audio_features_processed, audio_features_processed, audio_features_processed, None)
        # 合并视觉和听觉特征
        combined_features = torch.cat((visual_features, audio_features_processed), dim=2)
        # print("Combined features shape:", combined_features.size())
        current_batch_size = combined_features.size(0)

        # 决定是否更新记忆
        update_memory_decision = torch.sigmoid(self.update_memory_decision(combined_features))
        # 使用 torch.any() 来检查是否有任何元素大于 0.5
        should_update_memory = torch.any(update_memory_decision > 0.5)

        # 如果决定更新记忆，或者memory是None（第一次调用时）
        if memory is None or should_update_memory:
            # 确保输入特征的批次大小为1
            input_to_lstm = combined_features[:, 0, :]
            # 初始化记忆状态，如果它们是None或者需要更新
            if memory is None:
                memory = (torch.zeros(1, 256), torch.zeros(1, 256))
            # 更新记忆状态
            if current_batch_size==1:
                memory = self.memory_cell(input_to_lstm, memory)
                # print("new memory size",memory[0].size())



        # 将记忆状态扩展到与combined_features相同的批次大小
        memory_expanded=memory[0]
        # print("memory_expanded size",memory_expanded.size())
        memory_expanded = memory_expanded.unsqueeze(0).repeat(current_batch_size, 1, 1)

        combined_input = torch.cat((combined_features, memory_expanded), dim=2)

        # 计算动作概率

        action_probs = self.action_net(combined_input)
        return action_probs, memory


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
    Rewardaudio_data = sr.AudioData(Rewardaudio_data, 44100, sample_width=2)
    # 初始化语音识别器
    recognizer = sr.Recognizer()

    # 使用语音识别API将音频数据转换为文本
    try:
        #text = recognizer.recognize_google(Rewardaudio_data, language='en-US')
        text="well done"
        print("Recognized speech: ", text)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return 0.0  # 如果无法识别,则返回0作为奖励
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; ", e)
        return 0.0  # 如果请求失败,则返回0作为奖励

    # 解析文本以识别表扬
    if "well done" in text.lower() or "good job" in text.lower():
        return 1.0  # 正面奖励
    elif "not good" in text.lower() or "wrong" in text.lower():
        return -1.0  # 负面奖励
    else:
        return 0.0  # 无奖励




def get_Video():
    ret, frame = cap.read()  # 确保cap是一个已经打开的视频流
    if not ret:
        raise ValueError("无法从摄像头读取数据")
    # 将BGR图像转换为RGB格式并进行归一化
    video_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_frame = video_frame.astype(np.float32) / 255.0

    # 调整大小和归一化
    video_frame = cv2.resize(video_frame, (224, 224))
    video_frame = np.transpose(video_frame, (2, 0, 1))

    # 确保视频数据的形状是 [1, channels, height, width]
    video_data= torch.tensor(video_frame).unsqueeze(0).float()
    return video_data
def get_audio():
    # 尝试读取音频数据，添加异常处理
    try:
        audio_stream = pyaudio.PyAudio().open(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100,
            input=True,
            frames_per_buffer=1024  # 保持原来的设置
        )
        audio_data = audio_stream.read(1024)  # 确保audio_stream是一个已经打开的音频流
        audio_stream.stop_stream()
        audio_stream.close()  # 终止PyAudio实例
    except OSError as e:
        if e.errno == -9981:
            print("音频输入溢出，重置音频流...")
            # 关闭当前音频流
            # 创建新的音频流
            audio_stream = pyaudio.PyAudio().open(
                format=pyaudio.paInt16,
                channels=1,
                rate=44100,
                input=True,
                frames_per_buffer=1024  # 保持原来的设置
            )
            # 重新尝试读取音频数据
            audio_data = audio_stream.read(1024)
        else:
            raise  # 抛出其他类型的异常
    # 检查读取的音频数据长度
    if len(audio_data) < 1024:
        # 如果读取的数据不足1024字节，填充剩余的部分
        audio_data += b'\x00' * (1024 - len(audio_data))
    # 检查读取的音频数据长度
    if len(audio_data) > 1024:
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
    audio_data=audio_data.clone().detach().float()
    return audio_data
# 初始化摄像头和麦克风
cap = cv2.VideoCapture(0)

# 初始化模型
model = ComplexMultiModalNN()

# 初始化优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 初始化环境
env = RobotEnv()

# 初始化记忆状态
memory = None # LSTM的隐藏状态和细胞状态
# 在训练循环的开始处设置 epsilon
epsilon = 0.1  # 例如，设置为0.1
# 训练循环

num_episodes = 20  # 设置一个较大的episode数,以便模型可以持续学习
for episode in range(num_episodes):
    # 重置环境和记忆
    replay_buffer = ReplayBuffer(capacity=10000)
    state,audio_state = env.reset()

    total_reward = 0
    print("开始循环训练：", episode)
    # 执行动作并收集经验
    total_timestip=10
    buffer_count=0
    for t in range(total_timestip):  # 假设每个episode有1000个时间步

        print("时间步：", t)
        # 选择动作
        model.eval()
        with torch.no_grad():

            video_tensor =get_Video()
            audio_tensor = get_audio()
            action_probs, memory = model(video_tensor, audio_tensor, memory)
        action = torch.argmax(action_probs).item()

        result = env.step(action)
        if result is not None:
            next_state, next_audio_state,reward, done, info = result

            replay_buffer.push(state, action, reward, next_state, audio_state,next_audio_state,done)
            state = next_state
            audio_state=next_audio_state
            total_reward += reward
            buffer_count=buffer_count+1
        else:
            # 处理返回值为None的情况
            print("env.step returned None, which is not iterable.")


        # 如果是最后一个时间步，播放询问音频
        if t == total_timestip-1:
            print("我干得好吗？此处已经注释")
            #question_audio = env.text_to_speech("我干得好吗？")
            #env.play_audio(question_audio)

        # 如果episode结束，跳出循环
        #if done:
            #break

    # 训练模型
    print("实施训练")
    model.train()
    for _ in range(buffer_count):  # 每个episode训练buffer_count次

        states, actions, rewards, next_states, audio_states,next_audio_states,dones = replay_buffer.sample(batch_size=6)
        # 确保 dones 是一个一维的布尔张量


        # 计算目标 Q 值
        q_cal=torch.max(model(next_states,next_audio_states,memory)[0], dim=2)[0]
        target_q_values = rewards + (1 - dones) * 0.99 * torch.max(model(next_states,next_audio_states,memory)[0], dim=2)[0]

        # 计算当前 Q 值
        current_q_values=model(states, audio_states,memory)[0].gather(2, actions.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(current_q_values, target_q_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 输出信息
    print("完成一次训练，已保存模型")

    # 保存模型
    torch.save(model.state_dict(), 'robot_model.pt')
# 在所有episode完成后，执行资源释放
cap.release()  # 释放摄像头资源

if __name__ == '__main__':
    # ...省略初始化代码...
    # ...省略训练循环...
    # 在程序的最后，执行资源释放
    cap.release()

