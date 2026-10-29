import React, { useState, useEffect } from 'react';
import { 
  BrainCircuit, 
  Play, 
  Pause, 
  Square, 
  Save, 
  Terminal, 
  Activity, 
  CheckCircle2, 
  Sliders, 
  Sparkles,
  Cpu
} from 'lucide-react';
import { PageHeader } from '../components/layout/PageHeader';
import { Card } from '../components/common/Card';
import { Button } from '../components/common/Button';
import { Badge } from '../components/common/Badge';
import { TrainingTelemetry } from '../components/charts/TrainingTelemetry';
import { ApiService } from '../services/api';
import { TrainingSession, TrainingHyperparameters } from '../types';
import { useToast } from '../context/ToastContext';

export const ModelTrainingPage: React.FC = () => {
  const { showToast } = useToast();
  const [session, setSession] = useState<TrainingSession | null>(null);
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  // Hyperparameters
  const [config, setConfig] = useState<TrainingHyperparameters>({
    modelArchitecture: 'ChangeFormer (Transformer)',
    backbone: 'MiT-B2',
    epochs: 50,
    batchSize: 16,
    learningRate: 0.0001,
    optimizer: 'AdamW',
    lossFunction: 'Combined (Focal + Dice)',
    weightDecay: 0.01,
    validationSplit: 0.2,
    useDataAugmentation: true,
    cropSize: 512
  });

  useEffect(() => {
    const fetchSession = async () => {
      try {
        setIsLoading(true);
        const data = await ApiService.getTrainingSession();
        setSession(data);
        setConfig(data.config);
      } catch (err) {
        console.error('Error fetching training session:', err);
      } finally {
        setIsLoading(false);
      }
    };
    fetchSession();
  }, []);

  const handleTogglePause = () => {
    setIsPaused(!isPaused);
    showToast({
      type: 'info',
      title: isPaused ? 'Training Resumed' : 'Training Paused',
      message: `Worker node GPU-01 ${isPaused ? 'resumed backward pass' : 'paused at current epoch boundary'}.`
    });
  };

  const handleSaveCheckpoint = () => {
    showToast({
      type: 'success',
      title: 'Checkpoint Saved',
      message: `Model weights persisted to storage: checkpoints/changeformer_epoch_${session?.currentEpoch}.pth`
    });
  };

  const handleStartNewRun = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const newSession = await ApiService.startTrainingRun(config);
      setSession(newSession);
      showToast({
        type: 'success',
        title: 'Training Run Initialized',
        message: `Starting ${config.modelArchitecture} with ${config.backbone} backbone.`
      });
    } catch (err: any) {
      showToast({
        type: 'error',
        title: 'Training Failed to Start',
        message: err.message
      });
    }
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Deep Learning Sprawl Model Training"
        subtitle="Configure, fine-tune, and monitor multi-temporal transformer & convolutional change detection models in real time."
        breadcrumbs={[
          { label: 'Workstation', to: '/dashboard' },
          { label: 'Model Training' }
        ]}
        actions={
          <div className="flex items-center gap-2">
            <Button
              variant={isPaused ? 'emerald' : 'secondary'}
              size="sm"
              leftIcon={isPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
              onClick={handleTogglePause}
            >
              {isPaused ? 'Resume Run' : 'Pause Run'}
            </Button>
            <Button
              variant="outline"
              size="sm"
              leftIcon={<Save className="w-4 h-4" />}
              onClick={handleSaveCheckpoint}
            >
              Save Checkpoint
            </Button>
          </div>
        }
      />

      {/* Active Run Telemetry Summary */}
      {session && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <Card className="p-4">
            <div className="text-[11px] font-medium uppercase text-slate-400">Current Epoch</div>
            <div className="text-2xl font-bold font-mono text-cyan-400 mt-1">
              {session.currentEpoch} <span className="text-xs text-slate-500 font-normal">/ {session.totalEpochs}</span>
            </div>
            <div className="text-xs text-slate-400 mt-1 flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              <span>{Math.round((session.currentEpoch / session.totalEpochs) * 100)}% Complete</span>
            </div>
          </Card>

          <Card className="p-4">
            <div className="text-[11px] font-medium uppercase text-slate-400">Best mIoU (Mean IoU)</div>
            <div className="text-2xl font-bold font-mono text-emerald-400 mt-1">
              {session.bestMIoU}%
            </div>
            <div className="text-xs text-emerald-400 mt-1">
              +14.4% vs Baseline UNet
            </div>
          </Card>

          <Card className="p-4">
            <div className="text-[11px] font-medium uppercase text-slate-400">Best F1-Score</div>
            <div className="text-2xl font-bold font-mono text-white mt-1">
              {session.bestF1Score}%
            </div>
            <div className="text-xs text-slate-400 mt-1">
              Pixel-level change F1
            </div>
          </Card>

          <Card className="p-4">
            <div className="text-[11px] font-medium uppercase text-slate-400">GPU-01 Telemetry</div>
            <div className="text-2xl font-bold font-mono text-indigo-400 mt-1">
              14.2 <span className="text-xs text-slate-500 font-normal">GB / 24 GB</span>
            </div>
            <div className="text-xs text-slate-400 mt-1">
              NVIDIA A100 &bull; 58s / epoch
            </div>
          </Card>
        </div>
      )}

      {/* Main Realtime Telemetry Graph */}
      <Card className="space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan-400" />
            <h3 className="text-sm font-semibold text-white">Live Training & Validation Telemetry Curves</h3>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="processing" size="sm">Live Feed (Simulated)</Badge>
          </div>
        </div>

        {session && (
          <TrainingTelemetry data={session.history} />
        )}
      </Card>

      {/* Two Column Layout: Configuration Form + Live Epoch Logs */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Hyperparameter Config Form */}
        <Card className="space-y-4">
          <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
            <Sliders className="w-4 h-4 text-cyan-400" />
            <h3 className="text-sm font-semibold text-white">Hyperparameter Tuning</h3>
          </div>

          <form onSubmit={handleStartNewRun} className="space-y-4 text-xs">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-slate-300 block mb-1">Architecture</label>
                <select
                  value={config.modelArchitecture}
                  onChange={e => setConfig({ ...config, modelArchitecture: e.target.value as any })}
                  className="w-full bg-[#080c14] border border-slate-700 rounded-lg p-2 text-white"
                >
                  <option value="ChangeFormer (Transformer)">ChangeFormer (Transformer)</option>
                  <option value="BIT-CD (Bitemporal Trans)">BIT-CD (Bitemporal Transformer)</option>
                  <option value="Siam-UNet (Siamese CNN)">Siam-UNet (Siamese CNN)</option>
                </select>
              </div>

              <div>
                <label className="text-slate-300 block mb-1">Backbone Encoder</label>
                <select
                  value={config.backbone}
                  onChange={e => setConfig({ ...config, backbone: e.target.value as any })}
                  className="w-full bg-[#080c14] border border-slate-700 rounded-lg p-2 text-white"
                >
                  <option value="MiT-B2">MiT-B2 (SegFormer Hierarchical)</option>
                  <option value="Swin-Transformer-Base">Swin-Transformer-Base</option>
                  <option value="ResNet-50">ResNet-50 (Residual CNN)</option>
                  <option value="EfficientNet-B4">EfficientNet-B4</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-3">
              <div>
                <label className="text-slate-300 block mb-1">Epochs</label>
                <input
                  type="number"
                  value={config.epochs}
                  onChange={e => setConfig({ ...config, epochs: parseInt(e.target.value) })}
                  className="w-full bg-[#080c14] border border-slate-700 rounded-lg p-2 text-white font-mono"
                />
              </div>
              <div>
                <label className="text-slate-300 block mb-1">Batch Size</label>
                <input
                  type="number"
                  value={config.batchSize}
                  onChange={e => setConfig({ ...config, batchSize: parseInt(e.target.value) })}
                  className="w-full bg-[#080c14] border border-slate-700 rounded-lg p-2 text-white font-mono"
                />
              </div>
              <div>
                <label className="text-slate-300 block mb-1">Learning Rate</label>
                <input
                  type="text"
                  value={config.learningRate}
                  onChange={e => setConfig({ ...config, learningRate: parseFloat(e.target.value) })}
                  className="w-full bg-[#080c14] border border-slate-700 rounded-lg p-2 text-white font-mono"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-slate-300 block mb-1">Optimizer</label>
                <select
                  value={config.optimizer}
                  onChange={e => setConfig({ ...config, optimizer: e.target.value as any })}
                  className="w-full bg-[#080c14] border border-slate-700 rounded-lg p-2 text-white"
                >
                  <option value="AdamW">AdamW (Decoupled Weight Decay)</option>
                  <option value="Adam">Adam</option>
                  <option value="SGD + Momentum">SGD + Momentum</option>
                </select>
              </div>

              <div>
                <label className="text-slate-300 block mb-1">Loss Function</label>
                <select
                  value={config.lossFunction}
                  onChange={e => setConfig({ ...config, lossFunction: e.target.value as any })}
                  className="w-full bg-[#080c14] border border-slate-700 rounded-lg p-2 text-white"
                >
                  <option value="Combined (Focal + Dice)">Combined (Focal + Dice Loss)</option>
                  <option value="BCE + Dice Loss">BCE + Dice Loss</option>
                  <option value="Contrastive Loss">Contrastive Distance Loss</option>
                </select>
              </div>
            </div>

            <div className="pt-2">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={config.useDataAugmentation}
                  onChange={e => setConfig({ ...config, useDataAugmentation: e.target.checked })}
                  className="rounded bg-slate-900 border-slate-700 text-cyan-500"
                />
                <span className="text-slate-300">Enable Satellite Photometric & Spatial Augmentation (Flips, Solar Jitter)</span>
              </label>
            </div>

            <div className="pt-3 border-t border-slate-800 flex justify-end">
              <Button type="submit" variant="primary" size="md" leftIcon={<Play className="w-4 h-4" />}>
                Launch New Training Run
              </Button>
            </div>
          </form>
        </Card>

        {/* Live Epoch Console Output */}
        <Card className="space-y-3 flex flex-col justify-between">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div className="flex items-center gap-2">
              <Terminal className="w-4 h-4 text-emerald-400" />
              <h3 className="text-sm font-semibold text-white">Live PyTorch Worker Stream</h3>
            </div>
            <span className="text-[10px] font-mono text-slate-500">Node: gpu-worker-01 (CUDA 12.2)</span>
          </div>

          <div className="p-3 rounded-lg bg-[#05080f] border border-slate-800/80 font-mono text-[11px] text-slate-300 space-y-1.5 h-72 overflow-y-auto leading-relaxed">
            <div className="text-slate-500">[2024-10-21 18:42:01] Loading SpaceNet-7 train split: 3,424 pairs...</div>
            <div className="text-slate-500">[2024-10-21 18:42:15] Backbone: MiT-B2 weights initialized from ImageNet.</div>
            <div className="text-slate-400">[Epoch 01/50] train_loss: 0.684 | val_loss: 0.641 | mIoU: 52.4% | f1: 61.2% (58s)</div>
            <div className="text-slate-400">[Epoch 08/50] train_loss: 0.410 | val_loss: 0.395 | mIoU: 72.8% | f1: 79.5% (58s)</div>
            <div className="text-slate-400">[Epoch 16/50] train_loss: 0.264 | val_loss: 0.258 | mIoU: 81.9% | f1: 87.8% (57s)</div>
            <div className="text-slate-400">[Epoch 24/50] train_loss: 0.188 | val_loss: 0.191 | mIoU: 85.2% | f1: 91.2% (58s)</div>
            <div className="text-cyan-400">[Epoch 28/50] train_loss: 0.165 | val_loss: 0.174 | mIoU: 86.8% | f1: 92.4% (57s)</div>
            <div className="text-emerald-400">[SUCCESS] Best checkpoint updated: mIoU 86.8% &bull; Saved to disk.</div>
            <div className="text-slate-400 animate-pulse">&gt; Epoch 29/50: [========================&gt;......] 78% | Batch 168/214 | Loss: 0.158</div>
          </div>

          <div className="flex items-center justify-between text-[11px] text-slate-500 font-mono">
            <span>Learning Rate Decay: Cosine Annealing</span>
            <span>Batch Loss: 0.158</span>
          </div>
        </Card>
      </div>
    </div>
  );
};
