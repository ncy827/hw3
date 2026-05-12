from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from Gridworld import Gridworld
# 匯入 HW3-3 專用的 Lightning Agent
from agents_lightning import LightningDQNAgent as DQNAgent
import asyncio
import json
import numpy as np
import os

app = FastAPI()

# 確保 static 資料夾存在並掛載，這樣 index3.html 才能被讀取
if not os.path.exists("static"): os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

def get_pos_list(component):
    """處理 GridBoard 中組件座標提取"""
    if component is None: return []
    if isinstance(component, list):
        results = []
        for item in component:
            if hasattr(item, 'pos'):
                results.append(item.pos.tolist() if hasattr(item.pos, 'tolist') else list(item.pos))
        return results
    if hasattr(component, 'pos'):
        pos = component.pos.tolist() if hasattr(component.pos, 'tolist') else list(component.pos)
        return [pos]
    return []

@app.websocket("/ws/train")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket 連線成功！(HW3-3 專用後端)")
    
    try:
        raw_data = await websocket.receive_text()
        config = json.loads(raw_data)
        mode = config.get("mode", "random") # 預設改為 random
        algo = config.get("algo", "dueling")
        
        # 初始化 Lightning Agent
        agent = DQNAgent(input_dim=64, output_dim=4, mode=algo)

        for ep in range(2000): # Random 模式建議增加訓練回合數
            env = Gridworld(size=4, mode=mode)
            state = env.board.render_np().flatten().astype(float)
            done = False
            total_reward = 0
            step = 0
            
            while not done and step < 50:
                action_idx = agent.get_action(state)
                action_map = ['u', 'd', 'l', 'r']
                
                env.makeMove(action_map[action_idx])
                reward = env.reward()
                next_state = env.board.render_np().flatten().astype(float)
                
                done = True if reward == 10 or reward == -10 else False
                
                # 使用 Lightning Agent 的訓練介面
                agent.memory.append((state, action_idx, reward, next_state, done))
                agent.train_step() # 內部已包含 Huber Loss 與梯度裁剪
                
                state = next_state
                total_reward += reward
                step += 1

                try:
                    payload = {
                        "episode": ep,
                        "reward": int(total_reward),
                        "player": get_pos_list(env.board.components.get('Player'))[0] if get_pos_list(env.board.components.get('Player')) else [0,0],
                        "goal": get_pos_list(env.board.components.get('Goal'))[0] if get_pos_list(env.board.components.get('Goal')) else [0,0],
                        "pits": get_pos_list(env.board.components.get('Pit', [])),
                        "walls": get_pos_list(env.board.components.get('Wall', []))
                    }
                    await websocket.send_json(payload)
                except: pass
                
                await asyncio.sleep(0.005) # Random 模式建議加快動畫速度以利觀察長期趨勢
            
            if ep % 10 == 0:
                agent.update_target_network()
                print(f"HW3-3 Ep {ep} | Reward: {total_reward} | Eps: {agent.epsilon:.2f}")

    except Exception as e:
        print(f"後端錯誤: {e}")
    finally:
        print("連線關閉")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)