# GitHub Push 通道排障记录（本机 Windows / 国内网络）

> 记录时间：2026-09-23
> **状态：已解决。** 2026-09-23 10:31 成功推送 `2623593..dba5c42`，后续 `git push` 一条命令即可。
> 适用仓库：`momomingming/marine-antifouling-ml`（本地 `D:\锂硫电池\marine-antifouling-ml`）
> 本文价值：国内环境推 GitHub 反复失败时，按此顺序定位，不用从头试错。

---

## 零、最终可用配置（照抄即可，别再排查）

```bash
# 1) 镜像 host 的凭据桥接脚本 ~/.git-cred-mirror.sh
#    作用: 从 gh keyring 取真 token, 改写成镜像 host 返回
git config --global credential.helper '!sh C:/Users/HJ/.git-cred-mirror.sh'
git config --global credential."https://gh-proxy.com".helper '!sh C:/Users/HJ/.git-cred-mirror.sh'
git config --global credential."https://ghfast.top".helper   '!sh C:/Users/HJ/.git-cred-mirror.sh'

# 2) 通道: 让 github.com 自动走 gh-proxy.com(实测转发认证; ghfast.top 会剥掉认证)
git config --global url."https://gh-proxy.com/https://github.com/".insteadOf "https://github.com/"

# 3) 之后就是普通 push
git push origin main
```

### 三个致命坑（本次耗时的真正原因）

1. **GCM 弹 grafical 对话框阻塞**：Windows 凭据管理器会在后台弹窗等人点，
   非交互环境下表现为"push 卡 6~10 分钟无输出"，最后 `fatal: User cancelled dialog.`
   → 必须**把 `credential.helper` 换成自己的脚本**，彻底排除 GCM。
2. **`credential.helper ""` 会清空整条 helper 链**（含 URL-specific），
   导致 `could not read Username`。要置空就改用具体脚本值。
3. **git 的 URL 匹配 key 不能带尾斜杠**：`credential.https://gh-proxy.com.helper` ✅，
   `credential.https://gh-proxy.com/.helper` ❌（匹配不上，静默失效）。

---

## 一、现象的误导性

| 表象 | 真实原因 |
|---|---|
| `curl https://github.com/` 返回 HTTP=000 | 直连被墙（正常现象，**不代表不能推**） |
| `gh auth status` 有时报未登录 | keyring 在某些窗口读不到，**凭据本身仍在**，重试或换窗口可恢复 |
| `git push` 卡 6~9 分钟无任何输出 | **不是**网络挂死，多半是推送方的 credential helper 没签发密码 |
| `could not read Username for 'https://ghfast.top'` | helper 不认识镜像 host → 退化为匿名请求 → 被 GitHub 拒 |

**最容易踩的坑**：以为"通道不通"。实际上多数国产 GitHub 镜像的**读**和**写握手**都是通的，
卡的是**认证层**——镜像主机名和 `github.com` 不一致，原生 helper 不会给它签发凭据。

---

## 二、三段式定位法（按顺序做，别跳）

### 1. 读通道：能否拿到远端 SHA

```bash
timeout 40 git ls-remote origin main
```
拿到 SHA = 读通道 OK。**这步能过，就说明镜像可用，别急着换镜像。**

### 2. 写握手：能不能连上 receive-pack

```bash
curl -s -m 15 "https://ghfast.top/https://github.com/<owner>/<repo>.git/info/refs?service=git-receive-pack" \
  -H "User-Agent: git/2.0" | tr -d '\0' | head -c 60
```

- 返回 **`No anonymous write access.`** → ✅ 写通道正常，GitHub 在等你给凭据（**这是好消息**）
- 返回空 → 该镜像写通道不可用，换下一个

### 3. 认证：helper 到底有没有签发密码

```bash
printf 'protocol=https\nhost=<镜像host>\n\n' | git credential fill
```

- 输出里**有 password** → 认证 OK，push 应当能成功
- 只输出 username、**没有 password** → **根因在这**：helper 不认镜像 host

---

## 三、根因与修复（核心）

使用 `url.<镜像>/https://github.com/.insteadOf https://github.com/` 时，
git 拿 **重写后的 URL** 去问 credential helper，host 变成 `ghfast.top` 之类。
而 gh / GCM 只为 `github.com` 签发凭据 → 拿不到密码 → 匿名 push → 被拒或挂起。

### 修复：给镜像 host 挂一个"凭据改写" helper

创建 `~/.git-cred-mirror.sh`：

```sh
#!/bin/sh
set -e
target_host=""
while IFS='=' read -r k v; do
    [ "$k" = "host" ] && target_host="$v"
done
[ -z "$target_host" ] && exit 0

case "$1" in
  get)
    printf 'protocol=https\nhost=github.com\n\n' \
      | git credential fill 2>/dev/null \
      | sed -e "s|^host=.*|host=${target_host}|" \
      | grep -E '^(protocol|host|username|password)=' || true
    ;;
  *) exit 0 ;;
esac
```

注册（对所有要用的镜像逐个配）：

```bash
git config --global credential."https://ghfast.top/".helper   '!sh C:/Users/<你>/.git-cred-mirror.sh'
git config --global credential."https://gh-proxy.com/".helper '!sh C:/Users/<你>/.git-cred-mirror.sh'
```

token 全程只在管道里流转，不落盘、不进命令行历史。

### ⚠️ 已知副作用

`git credential fill` 内部会调 Windows 凭据管理器（manager-core）解封，
在本机**首次调用可能耗时 5~8 分钟**，期间看起来像卡死。**耐心等，别中断。**
后续调用会快很多。

---

## 四、可用镜像实测（2026-09-23）

| 镜像 | 根路径 | 读握手 | 写握手 |
|---|---|---|---|
| `ghfast.top` | HTTP=200 | ✅ | ✅ |
| `gh-proxy.com` | HTTP=000（根路径不通，不影响 git） | ✅ | ✅ |
| `gh-proxy.net` | HTTP=302 | ❌ 超时 | ❌ |
| `gh.idayer.com` | HTTP=000 | ❌ | ❌ |
| `mirror.ghproxy.com` | HTTP=000 | ❌ | ❌ |

> 教训：**根路径 HTTP=200 不代表 git 通，根路径 000 也不代表 git 不通。**
> 唯一可靠的判据是 `info/refs` 握手是否返回数据。

---

## 五、Plan B：完全绕开本地 push（当认证/网络彻底无解时）

服务器（`lhins-gkzr2xzg`）实测**能直连 GitHub 资源**：

- `https://ghfast.top/https://raw.githubusercontent.com/...` → 200
- `https://gh-proxy.com/https://codeload.github.com/.../zip/refs/heads/main` → 200

所以还有一条路：把本地产物发布成一个公网可下载链接
（WorkBuddy 的"发布为应用"或微云分享），再在服务器上
`Invoke-WebRequest` 下载 → 覆盖 `C:\apps\marine-af` → `nssm restart`。
缺点：绕过了 GitHub，仓库版本历史不更新，**只应在应急时使用**，后续补推对齐。

---

## 六、服务器部署备忘

- 应用目录：`C:\apps\marine-af`，**扁平结构，无 git**
- 服务名：`MarineAF8502`（8502 端口）
- 服务器 Python `C:\Python311\python.exe` 已装 sklearn 1.9.0 / xgboost 3.2.0 / rdkit
  → **26MB 的 `models/logo_ensemble/` 可在服务器本地训练生成，不必上传**
- 每次 `git pull` 后记得核对 nssm `AppParameters` 指向的脚本路径
