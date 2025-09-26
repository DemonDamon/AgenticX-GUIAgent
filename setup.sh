#!/bin/bash
# AgenticX-GUIAgent 自动化环境搭建脚本
# 使用方法: bash setup.sh

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查命令是否存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 检查conda是否安装
check_conda() {
    if command_exists conda; then
        print_success "Conda已安装: $(conda --version)"
        return 0
    else
        print_error "Conda未安装，请先安装Anaconda或Miniconda"
        print_info "下载地址: https://docs.conda.io/en/latest/miniconda.html"
        return 1
    fi
}

# 检查ADB是否安装
check_adb() {
    if command_exists adb; then
        print_success "ADB已安装: $(adb version | head -n1)"
        return 0
    else
        print_error "ADB未安装，请先安装Android SDK Platform Tools"
        print_info "macOS: brew install android-platform-tools"
        print_info "Ubuntu: sudo apt install android-tools-adb"
        return 1
    fi
}

# 检查Python版本
check_python() {
    if command_exists python3; then
        python_version=$(python3 --version | cut -d' ' -f2)
        print_success "Python已安装: $python_version"
        return 0
    else
        print_error "Python3未安装"
        return 1
    fi
}

# 创建conda环境
create_conda_env() {
    print_info "创建conda环境: agenticx-guiagent"
    
    # 检查环境是否已存在
    if conda env list | grep -q "agenticx-guiagent"; then
        print_warning "环境agenticx-guiagent已存在，是否重新创建? (y/N)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            print_info "删除现有环境..."
            conda env remove -n agenticx-guiagent -y
        else
            print_info "使用现有环境"
            return 0
        fi
    fi
    
    conda create -n agenticx-guiagent python=3.9 -y
    print_success "Conda环境创建成功"
}

# 激活conda环境
activate_conda_env() {
    print_info "激活conda环境"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate agenticx-guiagent
    print_success "环境已激活: $(which python)"
}

# 安装AgenticX框架
install_agenticx() {
    print_info "安装AgenticX框架"
    
    # 获取AgenticX根目录
    AGENTICX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
    
    if [ -f "$AGENTICX_ROOT/setup.py" ] || [ -f "$AGENTICX_ROOT/pyproject.toml" ]; then
        print_info "在开发模式下安装AgenticX: $AGENTICX_ROOT"
        pip install -e "$AGENTICX_ROOT"
        print_success "AgenticX安装成功"
    else
        print_error "未找到AgenticX安装文件，请检查路径: $AGENTICX_ROOT"
        return 1
    fi
}

# 安装项目依赖
install_dependencies() {
    print_info "安装项目依赖"
    
    # 更新pip
    pip install --upgrade pip
    
    # 安装requirements.txt中的依赖
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "基础依赖安装完成"
    else
        print_error "未找到requirements.txt文件"
        return 1
    fi
    
    # 安装额外的移动设备控制工具
    print_info "安装移动设备控制工具"
    pip install adbutils pure-python-adb
    
    # 安装多模态模型支持
    print_info "安装多模态模型支持"
    pip install openai-clip
    
    print_success "所有依赖安装完成"
}

# 配置环境变量
setup_env_vars() {
    print_info "配置环境变量"
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_success "已创建.env文件"
            print_warning "请编辑.env文件，配置您的API密钥"
            print_info "主要需要配置:"
            print_info "  - LLM_PROVIDER (openai/deepseek/kimi等)"
            print_info "  - 对应的API_KEY"
        else
            print_error "未找到.env.example文件"
            return 1
        fi
    else
        print_info ".env文件已存在"
    fi
}

# 验证ADB连接
verify_adb_connection() {
    print_info "验证ADB连接"
    
    # 启动ADB服务
    adb start-server
    
    # 检查连接的设备
    devices=$(adb devices | grep -v "List of devices" | grep "device$" | wc -l)
    
    if [ "$devices" -gt 0 ]; then
        print_success "检测到 $devices 个已连接的设备"
        adb devices
    else
        print_warning "未检测到已连接的Android设备"
        print_info "请确保:"
        print_info "  1. 设备已通过USB连接到电脑"
        print_info "  2. 设备已启用USB调试"
        print_info "  3. 已在设备上授权USB调试"
    fi
}

# 验证安装
verify_installation() {
    print_info "验证安装"
    
    # 测试AgenticX导入
    if python -c "import agenticx; print('AgenticX导入成功')" 2>/dev/null; then
        print_success "AgenticX验证通过"
    else
        print_error "AgenticX导入失败"
        return 1
    fi
    
    # 测试主要依赖
    dependencies=("aiohttp" "appium" "opencv-python" "torch" "transformers" "pydantic")
    
    for dep in "${dependencies[@]}"; do
        if python -c "import ${dep//-/_}" 2>/dev/null; then
            print_success "$dep 验证通过"
        else
            print_warning "$dep 导入失败，可能需要重新安装"
        fi
    done
}

# 创建启动脚本
create_run_script() {
    print_info "创建启动脚本"
    
    cat > run.sh << 'EOF'
#!/bin/bash
# AgenticX-GUIAgent 启动脚本

# 激活conda环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agenticx-guiagent

# 检查环境变量
if [ ! -f ".env" ]; then
    echo "错误: 未找到.env文件，请先配置环境变量"
    exit 1
fi

# 检查ADB连接
if ! adb devices | grep -q "device$"; then
    echo "警告: 未检测到已连接的Android设备"
fi

# 启动系统
echo "启动AgenticX-GUIAgent系统..."
python main.py "$@"
EOF

    chmod +x run.sh
    print_success "启动脚本已创建: run.sh"
}

# 显示使用说明
show_usage() {
    print_success "\n🎉 AgenticX-GUIAgent环境搭建完成！"
    print_info "\n📋 接下来的步骤:"
    print_info "\n1. 配置环境变量:"
    print_info "   nano .env  # 编辑API密钥等配置"
    print_info "\n2. 连接Android设备:"
    print_info "   - 启用开发者选项和USB调试"
    print_info "   - 通过USB连接设备"
    print_info "   - 在设备上授权USB调试"
    print_info "\n3. 启动系统:"
    print_info "   ./run.sh --interactive  # 交互模式"
    print_info "   ./run.sh --task \"帮我发微信给jennifer\"  # 执行任务"
    print_info "\n4. 测试连接:"
    print_info "   adb devices  # 检查设备连接"
    print_info "\n📚 更多信息请查看: SETUP_GUIDE.md"
}

# 主函数
main() {
    print_info "🚀 开始AgenticX-GUIAgent环境搭建"
    print_info "项目路径: $(pwd)"
    
    # 系统检查
    print_info "\n📋 系统环境检查"
    check_conda || exit 1
    check_adb || exit 1
    check_python || exit 1
    
    # 环境搭建
    print_info "\n🔧 环境搭建"
    create_conda_env
    activate_conda_env
    install_agenticx
    install_dependencies
    setup_env_vars
    
    # 验证和配置
    print_info "\n✅ 验证配置"
    verify_adb_connection
    verify_installation
    create_run_script
    
    # 显示使用说明
    show_usage
}

# 错误处理
trap 'print_error "安装过程中发生错误，请检查上面的错误信息"' ERR

# 运行主函数
main "$@"