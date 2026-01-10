
# .devcontainer の説明

このフォルダはこのプロジェクト用の VS Code Dev Container 設定を置く場所です。<br>
コンテナを使うことで、開発環境（ツール、依存関係、ランタイム）をチームで統一できます。

- **使い方**:<br>
    VS Code でリポジトリを開き、コマンドパレットで `Dev Containers: Rebuild and Reopen in Container` を実行してください。これによりコンテナがビルドされ、ワークスペースがコンテナ内で開きます。

- **前提条件**:
	- ホストに Docker がインストールされ、デーモンが起動していること。
	- VS Code に Remote Development（Dev Containers）拡張がインストールされていること。

- **ホストとファイル権限について**:
	- 本プロジェクトでは、`.devcontainer/Dockerfile` と `.devcontainer/entrypoint.sh` にてホストの UID/GID に合わせる処理（ユーザ/グループの作成・変更や権限調整）を行っています。そのため通常は追加の UID/GID 設定は不要です。
	- 詳細は次のファイルを参照してください: [.devcontainer/Dockerfile](.devcontainer/Dockerfile) と [.devcontainer/entrypoint.sh](.devcontainer/entrypoint.sh)

- **Git 認証について**:
    - GitHub CLI (`gh`) を使用してリポジトリにアクセスする場合、ホスト側で `gh auth login` を実行して認証を完了させておく必要があります。
    - また、ホスト側で `git config --global user.name "Your Name"` と `git config --global user.email "your.email@address"` を実行して `.gitconfig` を作成しておいてください。
    - これらの設定が完了した状態で、`dev.containers.copyGitConfig` オプションを有効にすると、ホストの `.gitconfig` がコンテナ内にコピーされ、VS Code の Git 機能が正常に動作します。
    - ホストで認証が行われていないと、コンテナ内からのリモート操作に失敗することがあります。

