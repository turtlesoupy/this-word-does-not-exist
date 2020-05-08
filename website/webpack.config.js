const MiniCssExtractPlugin = require('mini-css-extract-plugin');
const TerserJSPlugin = require('terser-webpack-plugin');
const OptimizeCSSAssetsPlugin = require('optimize-css-assets-webpack-plugin');
const path = require('path');

module.exports = {
	optimization: {
		minimizer: [new TerserJSPlugin({}), new OptimizeCSSAssetsPlugin({})],
	},
	plugins: [
		new MiniCssExtractPlugin({
			filename: 'bundle.css',
			path: path.resolve(__dirname, 'static'),
		}),
	],
	entry: {
		main: ['./static_src/index.js', './static_src/index.css'],
	},
	output: {
		filename: 'bundle.js',
		path: path.resolve(__dirname, 'static'),
	},
	devtool: 'eval-source-map',
	module: {
		rules: [
			{
				test: /\.m?js$/,
				exclude: /(node_modules|bower_components)/,
				use: {
					loader: 'babel-loader',
					options: {
						presets: ['@babel/preset-env']
					}
				}
			},
			{
				test: /\.css$/,
				use: [
					{
						loader: MiniCssExtractPlugin.loader,
						options: {
							publicPath: './static',
						}
					},
					'css-loader',
				],
			},
		]
	}
};
