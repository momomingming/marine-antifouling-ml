# 海洋防污材料 ML 预测平台 — 部署状态（2026-09-14）

## 线上地址
**http://82.156.43.223:8502** （Gradio 平台，9 个标签页，文献库 639 篇）

## 服务器
- 腾讯云轻量应用服务器 Lighthouse：`lhins-gkzr2xzg`
- 公网 IP：`82.156.43.223`，地域 `ap-beijing-3`
- 系统：Windows Server，Python `C:\Python311`（3.11.9）
- 到期：2026-10-02（续费前提醒）

## 系统服务
- 服务名：**MarineAF**（nssm 托管，开机自启，进程崩溃自动重启）
- 启动命令：`C:\Python311\python.exe -u app.py`
- 工作目录：`C:\apps\marine-af`
- 环境变量：`GRADIO_SERVER_PORT=8502`（程序内部 `app.launch(server_name='0.0.0.0', server_port=8502)`，已绑定 0.0.0.0，无需改代码）
- nssm 路径：`C:\tools\nssm\nssm.exe`

常用运维命令（在服务器 PowerShell）：
```powershell
& C:\tools\nssm\nssm.exe restart MarineAF      # 重启
& C:\tools\nssm\nssm.exe stop MarineAF         # 停止
Get-Service MarineAF | Select-Object Name,Status
```

## 防火墙
- Lighthouse 控制台防火墙：已加 **TCP 8502** 入站规则（来源 0.0.0.0/0，动作允许）
- 云主机 Windows Defender 防火墙服务未运行（不挡端口，无需配置）

## Python 依赖（C:\Python311）
- 新增安装（部署时）：`gradio-6.27.0`、`rdkit-2026.3.6`、`lightgbm-4.7.0`、`shap-0.51.0`、`optuna-5.0.0`、`matplotlib-3.11.2`
- 复用 bf-platform 已装：`numpy-2.4.6`、`pandas-3.0.5`、`scikit-learn-1.9.0`、`xgboost-3.2.0`
- 国内源：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple`

⚠️ **关键坑（必看）**：`requirements.txt` 里写的是 `rdkit`，但装 `rdkit-pypi`（2022.9.5）会炸——
`AttributeError: _ARRAY_API not found`（rdkit-pypi 用 NumPy 1.x 编译，与云主机 NumPy 2.4.6 不兼容，
导致 `rdkit.Chem.Draw` / `ML.InfoTheory` import 失败，分子描述符与预测功能全废）。
**必须装新版官方 wheel `rdkit`（2026.3.6，兼容 NumPy 2）**。重装命令：
```powershell
C:\Python311\python.exe -m pip uninstall -y rdkit-pypi
C:\Python311\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple rdkit
```

## 运行产物说明
- 仓库 `玻尔比赛/hf_space/` 是平台本体（app.py / model.pkl 14MB / literature_db.csv 481KB / synthesis_routes.py）
- 服务器**只需 `literature_db.csv` 元数据**（481KB），无需 2.1GB 的 `pdfs/` PDF 全文
- 程序 `load_literature_db()` 优先读 `literature_db.pkl`，否则读同目录 `literature_db.csv`（639 篇）

## 更新流程（可持续迭代）
1. 本地改 `玻尔比赛/hf_space/` 下代码
2. 本地打包运行文件（排除 `__pycache__/.gradio/*.bak`）：
   ```bash
   python scripts/package_ma_deploy.py    # 产出 marine-antifouling-deploy.zip (~7MB)
   ```
3. 发 GitHub Release（走 api.github.com 通道，直连畅通）：
   ```bash
   gh release create vX.Y.Z -R momomingming/marine-antifouling-ml --latest marine-antifouling-deploy.zip
   ```
4. 云主机拉取覆盖（无 git，走 gh-proxy 镜像）：
   ```powershell
   & C:\tools\nssm\nssm.exe stop MarineAF
   Invoke-WebRequest -Uri 'https://gh-proxy.com/https://github.com/momomingming/marine-antifouling-ml/releases/download/vX.Y.Z/marine-antifouling-deploy.zip' -OutFile 'C:\temp\ma-deploy.zip' -UseBasicParsing
   Expand-Archive -Path C:\temp\ma-deploy.zip -DestinationPath C:\apps\marine-af -Force
   & C:\tools\nssm\nssm.exe start MarineAF
   ```
5. 等 ~10s，访问 http://82.156.43.223:8502 验证

## 与 bf-platform 的关系
- bf-platform（锂硫电池）：http://82.156.43.223:8501 ，服务 BFPlatform，同实例同服务器
- marine-antifouling：http://82.156.43.223:8502 ，服务 MarineAF，同实例同服务器
- 两者共用 `C:\Python311` 环境（依赖已并存在该解释器）
