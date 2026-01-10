
# .devcontainer の説明

このフォルダはこのプロジェクト用の VS Code Dev Container 設定を置く場所です。<br>
コンテナを使うことで、開発環境（ツール、依存関係、ランタイム）をチームで統一できます。

- **使い方**:<br>
    VS Code でリポジトリを開き、コマンドパレットで `Dev Containers: Rebuild and Reopen in Container` を実行してください。これによりコンテナがビルドされ、ワークスペースがコンテナ内で開きます。
- **前提条件**:
	- ホストに Docker がインストールされ、デーモンが起動していること。
	- VS Code に Remote Development（Dev Containers）拡張がインストールされていること。

- **重要な注意**:<br>
    コンテナ内でVSCodeのGit機能を利用するには、ホスト側の PC で事前に `gh auth login` を実行してログインしておいてください。<br>
    必ず、VSCode内でのログインだけでなく、コンソールからのログインを行ってください。<br>
    ホストで認証が行われていないと、コンテナ内からのリモート操作に失敗することがあります。

- **ホストとファイル権限について**:
	- 本プロジェクトでは、`.devcontainer/Dockerfile` と `.devcontainer/entrypoint.sh` にてホストの UID/GID に合わせる処理（ユーザ/グループの作成・変更や権限調整）を行っています。そのため通常は追加の UID/GID 設定は不要です。
	- 詳細は次のファイルを参照してください: [.devcontainer/Dockerfile](.devcontainer/Dockerfile) と [.devcontainer/entrypoint.sh](.devcontainer/entrypoint.sh)

必要であれば、ここにコンテナの Dockerfile、devcontainer.json の要点やポート、ボリュームの情報を追加します。

