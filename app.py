"""
围棋对战网页应用主文件
使用Flask和WebSocket实现
"""

import os
import csv
import uuid
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_socketio import SocketIO, emit, join_room, leave_room
from werkzeug.utils import secure_filename
from go_engine import GoEngine
from onnx_inference import ONNXGoPlayer
import threading
import time
import queue
import uuid

# 导入排行榜模块
from leaderboard import leaderboard

# 初始化Flask应用
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # 请在生产环境中更换为安全的密钥

# 初始化SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# 配置
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'onnx', 'data'}  # 允许上传.onnx和.data文件
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 用户数据和模型数据
users = {}  # 存储用户信息和上传的模型
battle_rooms = {}  # 存储对战房间信息
user_active_battles = {}  # 存储用户的活跃对局信息

# 对战任务队列
battle_queue = queue.Queue()
battle_workers = []  # 存储工作线程
MAX_CONCURRENT_BATTLES = 2  # 最大并发对战数

import json

# 读取账号密码
def load_users():
    """从TXT文件加载用户信息"""
    user_dict = {}
    try:
        # 尝试不同的编码方式读取TXT文件
        encodings = ['utf-8', 'gbk', 'gb2312']
        content = None
        used_encoding = None

        for encoding in encodings:
            try:
                with open('账号密码.txt', 'r', encoding=encoding) as f:
                    content = f.read()
                    used_encoding = encoding
                    print(f"Successfully read TXT file with encoding: {encoding}")  # Debug output
                    break
            except UnicodeDecodeError:
                print(f"Failed to read TXT file with encoding: {encoding}")  # Debug output
                continue

        if content is None:
            print("Failed to read TXT file with all attempted encodings")  # Debug output
            return user_dict

        # 按行分割内容
        lines = content.strip().split('\n')
        print(f"Total lines: {len(lines)}")  # Debug output

        # 跳过标题行
        for i, line in enumerate(lines):
            if i == 0:  # 跳过标题行
                print(f"Skipping header line: {line}")  # Debug output
                continue

            # 移除行尾的回车换行符
            line = line.strip()
            print(f"Processing line {i}: '{line}'")  # Debug output

            # 使用两个或更多空格作为分隔符分割
            # 文件格式: 姓名  用户名  密码
            parts = line.split()
            print(f"Parts: {parts}")  # Debug output

            # 检查行是否有足够的列
            if len(parts) < 3:
                print(f"Skipping invalid row with {len(parts)} fields: {parts}")  # Debug output
                continue

            # 提取姓名、用户名和密码
            # 注意：姓名中可能包含逗号，所以我们需要特殊处理
            name = parts[0]
            username = parts[1]
            password = parts[2]

            user_dict[username] = {
                'password': password,
                'name': name,
                'models': []  # 存储上传的模型路径
            }
            print(f"Loaded user: {username}, password: {password}, name: '{name}'")  # Debug output
    except Exception as e:
        print(f"Error loading users: {e}")
        import traceback
        traceback.print_exc()  # Debug output
    print(f"Total users loaded: {len(user_dict)}")  # Debug output

    # 加载用户模型信息
    load_user_models(user_dict)

    return user_dict

def load_user_models(user_dict):
    """从磁盘加载用户模型信息"""
    try:
        if os.path.exists('user_models.json'):
            with open('user_models.json', 'r', encoding='utf-8') as f:
                saved_models = json.load(f)
                for username, models in saved_models.items():
                    if username in user_dict:
                        user_dict[username]['models'] = models
                        print(f"Loaded models for {username}: {models}")  # Debug output
    except Exception as e:
        print(f"Error loading user models: {e}")

def save_user_models():
    """将用户模型信息保存到磁盘"""
    try:
        # 只保存用户名和模型路径
        models_to_save = {}
        for username, info in users.items():
            models_to_save[username] = info['models']

        with open('user_models.json', 'w', encoding='utf-8') as f:
            json.dump(models_to_save, f, ensure_ascii=False, indent=2)
        print("User models saved successfully")  # Debug output
    except Exception as e:
        print(f"Error saving user models: {e}")

def remove_model_from_leaderboard(model_path):
    """从排行榜中移除与指定模型相关的记录并重新计算积分"""
    try:
        # 从对战记录中移除相关记录
        records_to_remove = []
        for pair_key in leaderboard.battle_records.keys():
            models = pair_key.split(' vs ')
            if model_path in models:
                records_to_remove.append(pair_key)

        # 删除相关记录
        for pair_key in records_to_remove:
            print(f"Removing battle record: {pair_key}")
            del leaderboard.battle_records[pair_key]

        # 从模型积分中移除相关记录
        if model_path in leaderboard.model_scores:
            print(f"Removing model score for: {model_path}")
            del leaderboard.model_scores[model_path]

        # 重新计算所有模型的积分
        print("Recalculating all model scores...")
        recalculate_all_scores()

        # 保存更新后的排行榜数据
        leaderboard.save_data()

        print(f"Removed {len(records_to_remove)} battle records related to model: {model_path}")
    except Exception as e:
        print(f"Error removing model from leaderboard: {e}")

def recalculate_all_scores():
    """重新计算所有模型的积分"""
    try:
        # 清空现有积分
        leaderboard.model_scores.clear()

        # 根据现有的对战记录重新计算积分
        for pair_key, record in leaderboard.battle_records.items():
            models = pair_key.split(' vs ')
            model1_path = models[0]
            model2_path = models[1]
            wins = record['wins']
            losses = record['losses']

            # 计算积分（只有在总对局数小于4时才更新积分）
            total_battles = wins + losses
            if total_battles <= 4:
                if wins > losses:
                    # 模型1胜出
                    leaderboard.model_scores[model1_path] += 3
                    leaderboard.model_scores[model2_path] += 1
                elif losses > wins:
                    # 模型2胜出
                    leaderboard.model_scores[model1_path] += 1
                    leaderboard.model_scores[model2_path] += 3
                else:
                    # 平局
                    leaderboard.model_scores[model1_path] += 2
                    leaderboard.model_scores[model2_path] += 2

        print(f"Recalculated scores for {len(leaderboard.model_scores)} models")
    except Exception as e:
        print(f"Error recalculating scores: {e}")
        import traceback
        traceback.print_exc()

def battle_worker():
    """对战工作线程"""
    while True:
        try:
            # 从队列中获取对战任务
            room_id = battle_queue.get(timeout=1)
            if room_id is None:
                break

            print(f"Starting battle for room: {room_id}")
            # 执行对战
            run_battle(room_id)
            print(f"Finished battle for room: {room_id}")

            battle_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in battle worker: {e}")

# 初始化工作线程
def init_battle_workers():
    """初始化对战工作线程"""
    global battle_workers
    for i in range(MAX_CONCURRENT_BATTLES):
        worker = threading.Thread(target=battle_worker, daemon=True)
        worker.start()
        battle_workers.append(worker)
    print(f"Started {MAX_CONCURRENT_BATTLES} battle workers")

# 加载用户数据
users = load_users()
print(f"Users dict: {users}")  # Debug output

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """首页"""
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """用户登录"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        print(f"Login attempt - Username: {username}, Password: {password}")  # Debug output
        print(f"Users dict: {users}")  # Debug output
        print(f"Username in users: {username in users}")  # Debug output
        if username in users:
            print(f"Stored password: {users[username]['password']}")  # Debug output
            print(f"Password match: {users[username]['password'] == password}")  # Debug output

        # 验证用户
        if username in users and users[username]['password'] == password:
            session['username'] = username
            session['name'] = users[username]['name']
            print(f"Login successful for {username}")  # Debug output
            return jsonify({'success': True})
        else:
            print(f"Login failed for {username}")  # Debug output
            return jsonify({'success': False, 'message': '用户名或密码错误'})

    return render_template('login.html')

@app.route('/logout')
def logout():
    """用户登出"""
    session.pop('username', None)
    session.pop('name', None)
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    """用户仪表板"""
    print(f"Dashboard access - Session: {session}")  # Debug output
    if 'username' not in session:
        print("No username in session, redirecting to login")  # Debug output
        return redirect(url_for('login'))

    username = session['username']
    print(f"Dashboard access for user: {username}")  # Debug output

    if username not in users:
        print(f"User {username} not found in users dict")  # Debug output
        return redirect(url_for('login'))

    user_models = users[username]['models']
    print(f"User models: {user_models}")  # Debug output

    # 获取其他用户的模型用于对战
    other_models = []
    for user, info in users.items():
        if user != username:
            for model in info['models']:
                other_models.append({
                    'username': user,
                    'name': info['name'],
                    'model_path': model
                })
    print(f"Other models: {other_models}")  # Debug output

    return render_template('dashboard.html',
                          name=session['name'],
                          models=user_models,
                          other_models=other_models,
                          users=users,
                          session=session)

@app.route('/upload_model', methods=['POST'])
def upload_model():
    """上传模型"""
    if 'username' not in session:
        return jsonify({'success': False, 'message': '请先登录'})

    username = session['username']

    # 检查是否已达到模型上传上限
    if len(users[username]['models']) >= 3:
        return jsonify({'success': False, 'message': '已经达到三个模型上限,请对leaf的电脑好一点'})

    # 检查是否有文件上传
    if 'model_file' not in request.files:
        return jsonify({'success': False, 'message': '没有选择文件'})

    file = request.files['model_file']

    # 检查文件名
    if file.filename == '':
        return jsonify({'success': False, 'message': '没有选择文件'})

    # 检查文件类型和保存
    if file and allowed_file(file.filename):
        # 生成安全的文件名
        original_filename = secure_filename(file.filename)
        # 添加用户名前缀以避免冲突
        filename = f"{username}_{original_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # 保存文件
        file.save(filepath)

        # 特殊处理：如果是.onnx文件，检查是否有对应的.data文件需要上传
        if filename.endswith('.onnx'):
            # 检查请求中是否有对应的数据文件
            data_file_key = 'data_file'  # 前端需要提供这个字段
            if data_file_key in request.files:
                data_file = request.files[data_file_key]
                if data_file and data_file.filename != '' and allowed_file(data_file.filename):
                    # 保存.data文件，保持原始文件名
                    data_original_filename = secure_filename(data_file.filename)
                    data_filepath = os.path.join(app.config['UPLOAD_FOLDER'], data_original_filename)
                    data_file.save(data_filepath)
                    print(f"Saved data file as: {data_filepath}")

        # 添加到用户模型列表（只添加.onnx文件）
        if filename.endswith('.onnx'):
            users[username]['models'].append(filepath)

        # 保存用户模型信息
        save_user_models()

        return jsonify({'success': True, 'message': '模型上传成功', 'filename': filename})
    else:
        return jsonify({'success': False, 'message': '只允许上传.onnx和.data文件（.data文件为可选）'})

@app.route('/delete_model', methods=['POST'])
def delete_model():
    """删除模型"""
    if 'username' not in session:
        return jsonify({'success': False, 'message': '请先登录'})

    username = session['username']
    data = request.get_json()
    model_path = data.get('model_path')

    # 检查模型是否属于当前用户
    if model_path in users[username]['models']:
        try:
            # 删除主模型文件
            if os.path.exists(model_path):
                os.remove(model_path)

            # 删除相关的.data文件（如果存在）
            # 根据ONNX的规范，.data文件通常与.onnx文件同名
            data_file_path = model_path + '.data'
            if os.path.exists(data_file_path):
                os.remove(data_file_path)

            # 也尝试删除与原始文件名匹配的.data文件
            original_model_name = os.path.basename(model_path)
            if original_model_name.endswith('.onnx'):
                original_data_name = original_model_name[:-5] + '.onnx.data'  # 移除.onnx，添加.onnx.data
                original_data_path = os.path.join(os.path.dirname(model_path), original_data_name)
                if os.path.exists(original_data_path):
                    os.remove(original_data_path)
                    print(f"Deleted original format data file: {original_data_path}")

            # 从用户模型列表中移除
            users[username]['models'].remove(model_path)

            # 从排行榜中移除相关记录
            remove_model_from_leaderboard(model_path)

            # 保存用户模型信息
            save_user_models()

            return jsonify({'success': True, 'message': '模型删除成功'})
        except Exception as e:
            return jsonify({'success': False, 'message': f'删除失败: {str(e)}'})
    else:
        return jsonify({'success': False, 'message': '模型不存在或不属于您'})

@app.route('/get_user_models/<username>')
def get_user_models(username):
    """获取指定用户的模型列表"""
    if 'username' not in session:
        return jsonify({'success': False, 'message': '请先登录'})

    if username not in users:
        return jsonify({'success': False, 'message': '用户不存在'})

    # 获取用户的模型列表
    user_models = users[username]['models']
    model_list = []

    for model_path in user_models:
        # 提取文件名作为模型名称
        model_name = os.path.basename(model_path)
        model_list.append({
            'name': model_name,
            'path': model_path
        })

    return jsonify({'success': True, 'models': model_list})

@app.route('/get_active_battle')
def get_active_battle():
    """获取用户的活跃对局"""
    if 'username' not in session:
        print("No username in session")  # Debug output
        return jsonify({'success': False, 'message': '请先登录'})

    username = session['username']
    print(f"=== Checking active battle for user: {username} ===")  # Debug output
    print(f"User active battles dict: {user_active_battles}")  # Debug output

    # 检查用户是否有活跃对局
    if username in user_active_battles:
        room_id = user_active_battles[username]
        print(f"Found room_id in user_active_battles: {room_id}")  # Debug output
        print(f"Battle rooms dict: {battle_rooms}")  # Debug output
        # 检查对局是否仍然存在且正在进行
        if room_id in battle_rooms:
            print(f"Room exists in battle_rooms: {battle_rooms[room_id]}")  # Debug output
            if battle_rooms[room_id]['battle_active']:
                print(f"Active battle found: {room_id}")  # Debug output
                return jsonify({'success': True, 'room_id': room_id})
            else:
                print(f"Battle is not active: {battle_rooms[room_id]['battle_active']}")  # Debug output
        else:
            print(f"Room not found in battle_rooms")  # Debug output
            # 清理过期的活跃对局记录
            print(f"Cleaning up expired battle record for user: {username}")  # Debug output
            if username in user_active_battles:
                del user_active_battles[username]
    else:
        print(f"User {username} not found in user_active_battles")  # Debug output

    print(f"No active battle found for user: {username}")  # Debug output
    return jsonify({'success': False})

@app.route('/start_battle', methods=['POST'])
def start_battle():
    """开始对战"""
    if 'username' not in session:
        print("No username in session")  # Debug output
        return jsonify({'success': False, 'message': '请先登录'})

    username = session['username']
    print(f"=== Starting battle for user: {username} ===")  # Debug output

    # 检查用户是否已有活跃对局
    if username in user_active_battles:
        room_id = user_active_battles[username]
        print(f"User already has active battle: {room_id}")  # Debug output
        if room_id in battle_rooms and battle_rooms[room_id]['battle_active']:
            print("Battle is already active, returning error")  # Debug output
            return jsonify({'success': False, 'message': '您已有一场正在进行的对局，请先完成该对局'})
        else:
            print("Existing battle is not active, allowing new battle")  # Debug output

    data = request.get_json()
    my_model_path = data.get('my_model_path')
    opponent_model_path = data.get('opponent_model_path')
    opponent_username = data.get('opponent_username')

    print(f"My model path: {my_model_path}")  # Debug output
    print(f"Opponent model path: {opponent_model_path}")  # Debug output
    print(f"Opponent username: {opponent_username}")  # Debug output

    # 验证我的模型是否属于当前用户
    if not my_model_path or my_model_path not in users[username]['models']:
        print("Invalid or missing my model")  # Debug output
        return jsonify({'success': False, 'message': '请选择有效的我的模型'})

    # 验证对手模型是否存在
    if not opponent_model_path or not opponent_username:
        print("Missing opponent info")  # Debug output
        return jsonify({'success': False, 'message': '无效的对手信息'})

    opponent_exists = False
    for user, info in users.items():
        if user == opponent_username and opponent_model_path in info['models']:
            opponent_exists = True
            break

    if not opponent_exists:
        print("Opponent model not found")  # Debug output
        return jsonify({'success': False, 'message': '对手模型不存在'})

    # 创建对战房间
    room_id = str(uuid.uuid4())
    print(f"Creating new battle room: {room_id}")  # Debug output

    battle_rooms[room_id] = {
        'player1': username,
        'player1_model': my_model_path,
        'player2': opponent_username,
        'player2_model': opponent_model_path,
        'board_size': 19,
        'game_count': 0,
        'total_games': 3,  # 总局数
        'games_per_match': 5,  # 每局场数
        'moves_per_game': 400,  # 每场手数
        'scores': [0, 0],  # 大比分 [player1胜局数, player2胜局数]
        'current_game': 1,  # 当前局数
        'current_match': 1,  # 当前场数
        'match_scores': [0, 0],  # 当前局的小比分 [player1, player2]
        'engine': None,
        'players': [],  # 连接的客户端
        'battle_thread': None,
        'battle_active': True  # 创建时就设为True
    }

    # 记录用户的活跃对局
    user_active_battles[username] = room_id
    print(f"Recorded active battle for user {username}: {room_id}")  # Debug output
    print(f"Updated user_active_battles: {user_active_battles}")  # Debug output

    # 将对战任务加入队列
    print(f"=== Queuing battle for room: {room_id} ===")  # Debug output
    battle_queue.put(room_id)
    print(f"Battle queued for room: {room_id}")  # Debug output

    return jsonify({'success': True, 'room_id': room_id, 'message': '对战已加入队列，将在资源可用时开始'})

@app.route('/battle/<room_id>')
def battle_room(room_id):
    """对战房间页面"""
    if 'username' not in session:
        return redirect(url_for('login'))

    if room_id not in battle_rooms:
        return "对战房间不存在", 404

    return render_template('battle.html')

@app.route('/leaderboard')
def leaderboard_page():
    """排行榜页面"""
    if 'username' not in session:
        return redirect(url_for('login'))

    # 获取排行榜数据
    leaderboard_data = leaderboard.get_leaderboard()

    # 获取用户信息以便显示模型名称
    user_model_mapping = {}
    for username, info in users.items():
        for model_path in info['models']:
            user_model_mapping[model_path] = {
                'username': username,
                'name': info['name'],
                'model_name': os.path.basename(model_path)
            }

    return render_template('leaderboard.html',
                          leaderboard_data=leaderboard_data,
                          user_model_mapping=user_model_mapping,
                          enumerate=enumerate)

@app.route('/api/battle_records')
def api_battle_records():
    """获取所有对战记录的API接口"""
    if 'username' not in session:
        return jsonify({'success': False, 'message': '请先登录'})

    # 获取所有对战记录
    records = leaderboard.get_all_battle_records()

    # 整理记录格式
    formatted_records = []
    for pair_key, record in records.items():
        models = pair_key.split(' vs ')
        model1_path = models[0]
        model2_path = models[1]

        # 获取模型信息
        model1_info = None
        model2_info = None

        for username, info in users.items():
            for model_path in info['models']:
                if model_path == model1_path:
                    model1_info = {
                        'username': username,
                        'name': info['name'],
                        'model_name': os.path.basename(model_path)
                    }
                elif model_path == model2_path:
                    model2_info = {
                        'username': username,
                        'name': info['name'],
                        'model_name': os.path.basename(model_path)
                    }

        formatted_records.append({
            'model1': model1_info,
            'model2': model2_info,
            'wins': record['wins'],
            'losses': record['losses'],
            'total': record['wins'] + record['losses']
        })

    return jsonify({'success': True, 'records': formatted_records})

@socketio.on('join_battle')
def handle_join_battle(data):
    """加入对战房间"""
    room_id = data['room_id']
    username = session.get('username')
    
    if not username:
        emit('error', {'message': '请先登录'})
        return
    
    if room_id not in battle_rooms:
        emit('error', {'message': '对战房间不存在'})
        return
    
    # 加入房间
    join_room(room_id)
    
    # 添加到房间的观察者列表
    if 'players' not in battle_rooms[room_id]:
        battle_rooms[room_id]['players'] = []
    
    if request.sid not in battle_rooms[room_id]['players']:
        battle_rooms[room_id]['players'].append(request.sid)
    
    # 发送初始状态
    room = battle_rooms[room_id]
    emit('battle_status', {
        'game_number': room['current_game'],
        'total_games': room['total_games'],
        'scores': room['scores'],
        'board': [[0 for _ in range(room['board_size'])] for _ in range(room['board_size'])],
        'current_player': 1,
        'game_over': False,
        'winner': None
    }, room=request.sid)

@socketio.on('leave_battle')
def handle_leave_battle(data):
    """离开对战房间"""
    room_id = data['room_id']
    username = session.get('username')
    
    if not username or room_id not in battle_rooms:
        return
    
    # 离开房间
    leave_room(room_id)
    
    # 从观察者列表中移除
    if request.sid in battle_rooms[room_id]['players']:
        battle_rooms[room_id]['players'].remove(request.sid)

@socketio.on('start_battle')
def handle_start_battle(data):
    """开始对战"""
    room_id = data['room_id']
    username = session.get('username')

    if not username or room_id not in battle_rooms:
        emit('error', {'message': '对战房间不存在'})
        return

    room = battle_rooms[room_id]

    # 检查是否是房间创建者
    if room['player1'] != username:
        emit('error', {'message': '只有房间创建者可以开始对战'})
        return

    # 检查是否已经有对战在进行
    if room['battle_active']:
        emit('error', {'message': '对战已经在进行中'})
        return

    # 将对战任务加入队列
    print(f"=== Queuing battle for room: {room_id} ===")  # Debug output
    room['battle_active'] = True
    battle_queue.put(room_id)
    print(f"Battle queued for room: {room_id}")  # Debug output

    # 通知客户端对战已加入队列
    emit('battle_queued', {'message': '对战已加入队列，将在资源可用时开始'})
    print(f"Battle thread started for room: {room_id}")  # Debug output

def run_battle(room_id):
    """运行对战逻辑"""
    if room_id not in battle_rooms:
        return
    
    room = battle_rooms[room_id]
    
    # 加载模型
    try:
        player1 = ONNXGoPlayer(room['player1_model'], room['board_size'])
        player2 = ONNXGoPlayer(room['player2_model'], room['board_size'])
    except Exception as e:
        socketio.emit('error', {'message': f'模型加载失败: {str(e)}'}, room=room_id)
        room['battle_active'] = False
        return
    
    # 进行多局对战
    for game_num in range(1, room['total_games'] + 1):
        if not room['battle_active']:  # 检查是否应该停止
            break

        room['current_game'] = game_num
        room['match_scores'] = [0, 0]  # 重置当前局的小比分

        # 进行多场对战
        for match_num in range(1, room['games_per_match'] + 1):
            if not room['battle_active']:  # 检查是否应该停止
                break

            room['current_match'] = match_num
            room['engine'] = GoEngine(room['board_size'])

            # 发送游戏开始消息
            socketio.emit('game_start', {
                'game_number': game_num,
                'total_games': room['total_games'],
                'match_number': match_num,
                'total_matches': room['games_per_match'],
                'scores': room['scores'],  # 大比分
                'match_scores': room['match_scores'],  # 小比分
                'move_count': 0,  # 当前手数
                'max_moves': room['moves_per_game']  # 最大手数
            }, room=room_id)

            # 对局循环
            max_moves = room['moves_per_game']
            move_count = 0

            while not room['engine'].is_game_over() and move_count < max_moves and room['battle_active']:
                # 获取当前玩家
                current_player = player1 if room['engine'].current_player == 1 else player2

                # 选择落子
                try:
                    move = current_player.select_move(room['engine'], temperature=0.3)
                except Exception as e:
                    socketio.emit('error', {'message': f'模型预测出错: {str(e)}'}, room=room_id)
                    room['battle_active'] = False
                    return

                # 执行落子
                if room['engine'].make_move(move[0], move[1]):
                    move_count += 1

                    # 发送落子信息
                    socketio.emit('move_made', {
                        'row': move[0],
                        'col': move[1],
                        'player': 3 - room['engine'].current_player,  # 刚落子的玩家（切换前的玩家）
                        'board': room['engine'].board.tolist(),
                        'captured_stones': room['engine'].captured_stones,
                        'game_number': game_num,
                        'total_games': room['total_games'],
                        'match_number': match_num,
                        'total_matches': room['games_per_match'],
                        'scores': room['scores'],  # 大比分
                        'match_scores': room['match_scores'],  # 小比分
                        'move_count': move_count,  # 当前手数
                        'max_moves': max_moves  # 最大手数
                    }, room=room_id)

                    # 短暂延迟以控制速度（减少延迟以提高速度）
                    time.sleep(0.1)
                else:
                    # 无效落子，跳过
                    socketio.emit('error', {'message': f'无效落子: {move}'}, room=room_id)
                    break

            # 游戏结束，计算得分
            if room['battle_active']:
                black_score, white_score = room['engine'].simple_score()
                winner = 1 if black_score > white_score else 2

                if winner == 1:
                    room['match_scores'][0] += 1
                else:
                    room['match_scores'][1] += 1

                # 检查是否有人赢得当前局（先赢3场）
                if room['match_scores'][0] >= 3 or room['match_scores'][1] >= 3 or match_num == room['games_per_match']:
                    # 当前局结束，更新大比分
                    if room['match_scores'][0] > room['match_scores'][1]:
                        room['scores'][0] += 1
                    elif room['match_scores'][1] > room['match_scores'][0]:
                        room['scores'][1] += 1

                    # 发送局结束消息
                    socketio.emit('match_end', {
                        'winner': winner,
                        'black_score': black_score,
                        'white_score': white_score,
                        'scores': room['scores'],  # 大比分
                        'match_scores': room['match_scores'],  # 小比分
                        'game_number': game_num,
                        'total_games': room['total_games'],
                        'match_number': match_num,
                        'total_matches': room['games_per_match'],
                        'move_count': move_count,
                        'max_moves': max_moves
                    }, room=room_id)

                    # 检查是否有人赢得整个对战（先赢2局）
                    if room['scores'][0] >= 2 or room['scores'][1] >= 2:
                        # 对战结束，更新排行榜
                        final_winner = 1 if room['scores'][0] > room['scores'][1] else 2
                        player1_model = room['player1_model']
                        player2_model = room['player2_model']
                        player1_wins = (final_winner == 1)

                        # 更新排行榜
                        leaderboard.update_battle_result(player1_model, player2_model, player1_wins)
                        break  # 对战结束
                else:
                    # 发送场结束消息
                    socketio.emit('game_end', {
                        'winner': winner,
                        'black_score': black_score,
                        'white_score': white_score,
                        'scores': room['scores'],  # 大比分
                        'match_scores': room['match_scores'],  # 小比分
                        'game_number': game_num,
                        'total_games': room['total_games'],
                        'match_number': match_num,
                        'total_matches': room['games_per_match'],
                        'move_count': move_count,
                        'max_moves': max_moves
                    }, room=room_id)

                # 等待一段时间再开始下一场比赛（减少延迟以提高速度）
                time.sleep(1)

        # 检查是否有人赢得整个对战
        if room['scores'][0] >= 2 or room['scores'][1] >= 2:
            break  # 对战结束

    # 所有对战结束
    
    # 所有对战结束
    if room['battle_active']:
        final_winner = 1 if room['scores'][0] > room['scores'][1] else 2
        socketio.emit('battle_end', {
            'final_winner': final_winner,
            'scores': room['scores']
        }, room=room_id)

    print(f"=== Ending battle for room: {room_id} ===")  # Debug output
    room['battle_active'] = False
    print(f"Set battle_active to False for room: {room_id}")  # Debug output

    # 清理用户的活跃对局记录
    if room['player1'] in user_active_battles and user_active_battles[room['player1']] == room_id:
        print(f"Cleaning up active battle record for player1: {room['player1']}")  # Debug output
        del user_active_battles[room['player1']]
    if room['player2'] in user_active_battles and user_active_battles[room['player2']] == room_id:
        print(f"Cleaning up active battle record for player2: {room['player2']}")  # Debug output
        del user_active_battles[room['player2']]

    print(f"Updated user_active_battles: {user_active_battles}")  # Debug output

@socketio.on('disconnect')
def handle_disconnect():
    """处理客户端断开连接"""
    # 从所有房间的观察者列表中移除
    for room_id, room in battle_rooms.items():
        if request.sid in room.get('players', []):
            room['players'].remove(request.sid)

if __name__ == '__main__':
    # 初始化对战工作线程
    init_battle_workers()
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)