from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from Gridworld import Gridworld
from agents import DQNAgent
import asyncio
import json
import numpy as np
import os

app = FastAPI()

if not os.path.exists("static"): os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

def get_pos_list(component):
    """
    強健的座標提取函式：
    處理 GridBoard 中組件可能是單個物件或列表的情況
    """
    if component is None:
        return []
    
    # 如果已經是列表，遍歷提取座標
    if isinstance(component, list):
        results = []
        for item in component:
            if hasattr(item, 'pos'):
                results.append(item.pos.tolist() if hasattr(item.pos, 'tolist') else list(item.pos))
        return results
    
    # 如果是單個物件，直接提取座標並包裝成列表
    if hasattr(component, 'pos'):
        pos = component.pos.tolist() if hasattr(component.pos, 'tolist') else list(component.pos)
        return [pos]
    
    return []

@app.websocket("/ws/train")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket 連線成功！ (Codespaces)")
    
    try:
        raw_data = await websocket.receive_text()
        config = json.loads(raw_data)
        mode = config.get("mode", "static")
        algo = config.get("algo", "naive")
        
        # 初始化模型 (HW3-1: 64 維輸入, 4 維輸出)
        agent = DQNAgent(input_dim=64, output_dim=4, mode=algo)

        for ep in range(1000):
            env = Gridworld(size=4, mode=mode)
            state = env.board.render_np().flatten()
            done = False
            total_reward = 0
            step = 0
            
            while not done and step < 50:
                action_idx = agent.get_action(state)
                action_map = ['u', 'd', 'l', 'r']
                
                env.makeMove(action_map[action_idx])
                reward = env.reward()
                next_state = env.board.render_np().flatten()
                
                done = True if reward == 10 or reward == -10 else False
                
                # 訓練邏輯
                agent.memory.append((state, action_idx, reward, next_state, done))
                agent.train_step()
                
                state = next_state
                total_reward += reward
                step += 1

                # --- 數據發送部分 (已加入強健性修正) ---
                try:
                    # 提取各組件座標
                    player_pos_list = get_pos_list(env.board.components.get('Player'))
                    goal_pos_list = get_pos_list(env.board.components.get('Goal'))
                    pits_pos_list = get_pos_list(env.board.components.get('Pit', []))
                    walls_pos_list = get_pos_list(env.board.components.get('Wall', []))

                    payload = {
                        "episode": ep,
                        "reward": int(total_reward),
                        "player": player_pos_list[0] if player_pos_list else [0,0],
                        "goal": goal_pos_list[0] if goal_pos_list else [0,0],
                        "pits": pits_pos_list,
                        "walls": walls_pos_list
                    }
                    await websocket.send_json(payload)
                except Exception as send_err:
                    print(f"發送數據失敗: {send_err}")
                
                await asyncio.sleep(0.01) # 加快動畫速度
            
            if ep % 10 == 0:
                agent.update_target_network()
                print(f"Episode {ep} 完成，總獎勵: {total_reward}, 探索率: {agent.epsilon:.2f}")

    except Exception as e:
        print(f"後端發生錯誤: {e}")
    finally:
        print("連線關閉")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)