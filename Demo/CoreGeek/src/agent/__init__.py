"""CoreGeek 参赛 Agent 包。

模块划分（自上而下）：
    protocol.py  协议层：解析判题器的 Request、构造 Response 指令（纯数据，不含策略）
    grid.py      寻路层：八方向栅格 A*
    memory.py    记忆层：跨回合状态 + 基地几何布局规划
    brain.py     策略层：白天分工、自进化任务、BOSS 采购、夜晚集火
    server.py    接入层：HTTP 服务

数据流：server 收到 POST -> brain.decide() -> memory 先消化上一回合的动作合法性
        -> brain 决定本回合指令 -> 返回给 server 序列化成 JSON。
"""
