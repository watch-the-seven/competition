#!/usr/bin/env bash
# 判题器启动入口。接口文档第 1 章给出的样例就是：bash run.sh <port>
#
# $1 = 判题器为本队分配的监听端口。
# set -euo pipefail：任一命令失败、用到未定义变量、管道中任一环失败都直接退出，
# 避免带着错误状态继续跑。
set -euo pipefail

# 固定工作目录到本脚本所在目录（提交目录可能被解压到任意路径）。
cd "$(dirname "$0")"

# 用 exec 让 python 进程顶替当前 shell：
# 判题器 kill 掉这个 PID 时能直接终止 python，不会留下孤儿进程。
exec python3 main3.py "$1"
