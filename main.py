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
        states = [exp[0] for exp in experiences]
        actions = [exp[1] for exp in experiences]
        rewards = [exp[2] for exp in experiences]
        next_states = [exp[3] for exp in experiences]
        audio_states= [exp[4] for exp in experiences]
        next_audio_states = [exp[5] for exp in experiences]
        dones = [exp[6] for exp in experiences]
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
        self.memory = (torch.zeros(1, 128), torch.zeros(1, 128))  #
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
        # 在执行动作之前，确保 memory 已经被初始化
        if self.memory[0] is None:
            self.memory = (torch.zeros(1, 128), torch.zeros(1, 128))

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
        self.memory_cell = nn.LSTMCell(128, 128)  # 假设输入和隐藏状态的维度都是128

        # 初始化记忆更新决策层
        self.update_memory_decision = nn.Linear(128, 1)

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
        self.audio_fc = nn.Linear(65536, 128)  # 假设你需要将音频特征从1024维降到128维
        # 添加音频卷积层
        self.audio_conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.audio_conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        # 添加音频生成模块
        self.audio_generator = AudioGenerator(input_size=128, audio_sample_rate=44100, audio_length=16000)

        # 听觉处理部分
        self.fc_audio = nn.Linear(64, 128)

        # 自注意力层
        self.self_attention = SelfAttention(embed_size=128, heads=4)

        # 动作生成部分
        self.action_net = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # 假设有10种可能的动作
        )

    def forward(self, visual_input, audio_input, memory=None):

        if torch.cuda.is_available():
            visual_input = visual_input.cuda()
            audio_input = audio_input.cuda()



        # 视觉特征提取
        visual_input = F.relu(self.conv1(visual_input))
        visual_input = F.max_pool2d(visual_input, 2)
        visual_input = F.relu(self.conv2(visual_input))
        visual_input = F.max_pool2d(visual_input, 2)
        # 在forward方法中，卷积层之后添加打印语句
        print("Convolutional visual_input shape:", visual_input.size())
        # 计算卷积层输出的特征图的展平大小
        batch_size = visual_input.size(0)
        print("batch_size:", batch_size)
        num_features = visual_input.size(1) * visual_input.size(2) * visual_input.size(3)
        print("num_features:", num_features)
        # 展平特征图
        visual_input_flattened = visual_input.view(batch_size, -1)  # 展平为 [batch_size, num_features]
        print("visual_input_flattened:", visual_input_flattened)
        # 全连接层处理展平后的特征图
        visual_features = self.visual_fc(visual_input_flattened)

        # 听觉特征提取
        audio_features = F.relu(self.audio_conv1(audio_input))
        audio_features = F.relu(self.audio_conv2(audio_features))
        print("Audio features shape before flattening:", audio_features.size())
        # 此处应包含展平操作，假设音频特征在展平前的最后一维为1024
        # 展平音频特征，确保批次大小为1在最前面
        audio_features_flattened = audio_features.permute(0, 2, 1).contiguous().view(1, -1)
        print("Audio features shape after flattening:",
              audio_features_flattened.size())  # 应该输出 torch.Size([1, 1024])   print("Audio features shape after flattening:", audio_features_flattened.size())
        # assert audio_features_flattened.shape[1] == self.audio_fc.in_features
        audio_features_processed = self.audio_fc(audio_features_flattened)
        audio_features_processed = audio_features_processed.unsqueeze(1)
        # 自注意力
        visual_features = self.self_attention(visual_features, visual_features, visual_features, None)
        audio_features_processed= self.self_attention(audio_features_processed, audio_features_processed, audio_features_processed, None)
        # 合并视觉和听觉特征
        combined_features = torch.cat((visual_features, audio_features_processed), dim=1)
        print("Combined features shape:", combined_features.size())

        # 决定是否更新记忆
        update_memory_decision = torch.sigmoid(self.update_memory_decision(combined_features))
        # 使用 torch.any() 来检查是否有任何元素大于 0.5
        should_update_memory = torch.any(update_memory_decision > 0.5)
        # 如果决定更新记忆
        if memory is None or should_update_memory:
            # 更新长期记忆
            # 在 LSTM 单元的输入之前，确保提取正确的特征向量
            # 假设 combined_features 的第一个维度是批次大小，第二个维度是特征数量，第三个维度是特征向量的维度
            # 我们只需要第二个维度的特征向量
            input_to_lstm = combined_features[:, 0, :]  # 提取第一个特征向量，形状应为 [1, 128]

            # 现在将这个特征向量传递给 LSTM 单元
            memory = (self.memory_cell(input_to_lstm, memory)[0], memory[1])  # 只取隐藏状态
        # 假设 memory_expanded 的原始形状是 [1, 128]
        # 我们需要将其调整为 [1, 2, 128] 以匹配 combined_features 的第二个维度
        memory_expanded = memory[0].unsqueeze(1)  # 在第二个维度上增加一个维度
        print("memory_expanded:", memory_expanded.size())  # 现在应该是 [1, 1, 128]
        memory_expanded = memory_expanded.repeat(1, 2, 1)  # 重复第一个维度以匹配 combined_features 的第二个维度
        print("memory_expanded after repetition:", memory_expanded.size())  # 现在应该是 [1, 2, 128]
        # 连接后检查形状
        combined_input = torch.cat((combined_features, memory_expanded), dim=2)
        print("combined_input shape after concatenation:", combined_input.size())
        action_probs = self.action_net(combined_input)
        print("Action probabilities shape:", action_probs.size())
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


# 音频数据预处理
def process_audio(audio_data):
    # 这里可以添加音频数据的预处理步骤
    # 假设音频数据已经是归一化的,我们只需要将其转换为适合模型的格式
    # 假设模型期望的音频输入是固定长度的,例如1024个样本
    if isinstance(audio_data, torch.Tensor):
        print("audio_data is a PyTorch tensor.")
    else:
        # 使用struct.unpack处理音频数据
        audio_data = struct.unpack('b' * 1024, audio_data)
        audio_data = np.array(audio_data) / 32768.0
        # 将音频数据转换为适合卷积层的形状
        audio_data = np.reshape(audio_data, (1, 1, -1))  # [1, 1, sample_rate]
        audio_data = torch.tensor(audio_data, dtype=torch.float)
    return audio_data


# 视频数据预处理
def process_video(video_frame):
    # 将BGR图像转换为RGB格式并进行归一化
    video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
    video_frame = video_frame.astype(np.float32) / 255.0

    # 调整大小和归一化
    video_frame = cv2.resize(video_frame, (224, 224))
    video_frame = np.transpose(video_frame, (2, 0, 1))

    return video_frame
def get_Video():
    ret, frame = cap.read()  # 确保cap是一个已经打开的视频流
    if not ret:
        raise ValueError("无法从摄像头读取数据")

    # 预处理视频和音频数据
    processed_video = process_video(frame)
    # 确保视频数据的形状是 [1, channels, height, width]
    video_data= torch.tensor(processed_video).unsqueeze(0).float()
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
    audio_data=process_audio(audio_data)
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
memory = (torch.zeros(1, 128), torch.zeros(1, 128))  # LSTM的隐藏状态和细胞状态
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
        dones = torch.tensor(dones).float()

        # 计算目标 Q 值

        # 假设 rewards, next_states, next_audio_states, 和 dones 都是列表
        # 首先，初始化一个空列表来存储每个样本的目标 Q 值
        target_q_values_list = []

        # 遍历每个样本
        for i in range(len(rewards)):
            # 计算当前样本的目标 Q 值
            current_reward = rewards[i]
            current_next_state = next_states[i]
            current_next_audio_state = next_audio_states[i]
            current_done = dones[i]

            # 计算 max Q 值，如果模型接受单个样本作为输入
            # 假设 model 返回的是一个名为 action_probs 的张量
            action_probs, _ = model(current_next_state, current_next_audio_state, memory)
            # 使用 torch.max 获取最大值
            max_q_value = torch.max(action_probs, dim=1)[0]  # 获取最大值

            # 计算当前样本的目标 Q 值
            target_q_value = current_reward + (1 - current_done) * 0.99 * max_q_value

            # 将计算出的目标 Q 值添加到列表中
            target_q_values_list.append(target_q_value)

        # 将列表转换为张量
        target_q_values = torch.stack(target_q_values_list)
        print("Target Q values shape:", target_q_values.size())
        print("Target Q values:", target_q_values)

        # 计算当前 Q 值
        # 初始化 app_q_values 为一个空张量，形状为 [0, 10]
        app_q_values = torch.empty(0, 10)

        # 在循环中累积 Q 值张量
        for state, audio_state, action in zip(states, audio_states, actions):
            q_values, _ = model(state, audio_state, memory)

            # 选择第一个状态的所有动作的 Q 值，形状为 [10]
            current_q_values = q_values[0]  # 选择第一个状态的 Q 值

            # 如果是第一次迭代，初始化 app_q_values 为 current_q_values 的副本
            if app_q_values.nelement() == 0:
                app_q_values = current_q_values.unsqueeze(0)  # 添加批次维度
            else:
                # 否则，将 current_q_values 添加到 app_q_values 的下一个批次
                current_q_values = current_q_values.unsqueeze(0)  # 添加批次维度
                app_q_values = torch.cat((app_q_values, current_q_values), dim=0)

        # 现在 app_q_values 是一个包含所有时间步的 Q 值的张量，形状为 [T, 10]
        # 确保 actions_tensor 的形状是 [T, 1]
        actions_tensor = torch.tensor(actions).view(-1, 1)  # 确保 actions_tensor 是 2D 张量

        # 使用 gather 函数来提取特定动作的 Q 值
        current_q_values = app_q_values.gather(dim=1, index=actions_tensor).squeeze()
        # 打印 Q 值列表
        print("Current Q values:", current_q_values)

        print("Current Q values shape:", current_q_values.size())
        print("Current Q values:", current_q_values)
        loss = F.mse_loss(current_q_values, target_q_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 输出信息
    print("Episode {episode}: Total Reward {total_reward}")

    # 保存模型
    torch.save(model.state_dict(), 'robot_model.pt')
# 在所有episode完成后，执行资源释放
cap.release()  # 释放摄像头资源

if __name__ == '__main__':
    # ...省略初始化代码...
    # ...省略训练循环...
    # 在程序的最后，执行资源释放
    cap.release()

