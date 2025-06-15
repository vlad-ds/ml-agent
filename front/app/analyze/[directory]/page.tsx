"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { useEffect, useState, use } from "react"

interface AnalysisPageProps {
  params: Promise<{
    directory: string
  }>
}

interface AnalysisResult {
  task_completed: string
  best_model: string
  best_auc_score: number
  cv_auc_mean: number
  cv_auc_std: number
  model_rankings: [string, number][]
  top_features: string[]
  model_stability: string
  recommendation: string
}

export default function AnalysisPage({ params }: AnalysisPageProps) {
  const { directory } = use(params)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null)

  useEffect(() => {
    const analyzeData = async () => {
      try {
        setIsLoading(true)
        const response = await fetch('http://localhost:8000/model', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            prompt: `Train and evaluate models on the diabetes-readmission dataset using AUC as the evaluation metric`
          })
        })

        if (!response.ok) {
          throw new Error('Analysis failed')
        }

        console.log(response)

        const data = await response.json()
        
        console.log('API Response:', {
          status: data.status,
          result: {
            task_completed: data.result?.task_completed,
            best_model: data.result?.best_model,
            best_auc_score: data.result?.best_auc_score,
            cv_auc_mean: data.result?.cv_auc_mean,
            cv_auc_std: data.result?.cv_auc_std,
            model_rankings: data.result?.model_rankings,
            top_features: data.result?.top_features,
            model_stability: data.result?.model_stability,
            recommendation: data.result?.recommendation
          }
        })
        
        // Validate the response structure
        if (!data.status || data.status !== "success" || !data.result || 
            !data.result.task_completed || !data.result.best_model || 
            typeof data.result.best_auc_score !== 'number' ||
            !Array.isArray(data.result.model_rankings) ||
            !Array.isArray(data.result.top_features) ||
            !data.result.model_stability ||
            !data.result.recommendation) {
          console.error('Validation failed:', {
            status: data.status,
            hasResult: !!data.result,
            taskCompleted: !!data.result?.task_completed,
            bestModel: !!data.result?.best_model,
            bestAucScore: typeof data.result?.best_auc_score,
            modelRankings: Array.isArray(data.result?.model_rankings),
            topFeatures: Array.isArray(data.result?.top_features),
            modelStability: !!data.result?.model_stability,
            recommendation: !!data.result?.recommendation
          })
          throw new Error('Invalid response structure from API')
        }

        console.log('Validation passed, setting analysis result')
        setAnalysisResult(data.result)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred')
        console.error('Error details:', err)
      } finally {
        setIsLoading(false)
      }
    }

    analyzeData()
  }, [directory])

  return (
    <div className="flex flex-col min-h-screen justify-center items-center">
      <div className="container max-w-4xl py-10">
        <div className="space-y-8">
          <div className="text-center space-y-4">
            <h1 className="text-3xl font-bold">Dataset Analysis</h1>
            <p className="text-muted-foreground">
              Analyzing your dataset in directory: {directory}
            </p>
          </div>

          {isLoading && (
            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
                  <span className="ml-2">Analyzing your data...</span>
                </div>
              </CardContent>
            </Card>
          )}

          {error && (
            <Card className="border-red-200 bg-red-50">
              <CardContent className="p-6">
                <p className="text-red-600">Error: {error}</p>
                <p className="text-sm text-red-500 mt-2">
                  Please try again or contact support if the problem persists.
                </p>
              </CardContent>
            </Card>
          )}

          {analysisResult && (
            <div className="space-y-6">
              {/* Task Status */}
              <Card>
                <CardHeader>
                  <CardTitle>Task Status</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-lg">{analysisResult.task_completed}</p>
                </CardContent>
              </Card>

              {/* Model Performance */}
              <Card>
                <CardHeader>
                  <CardTitle>Model Performance</CardTitle>
                  <CardDescription>Results from model evaluation</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm font-medium">Best Model</p>
                      <p className="text-lg font-semibold">{analysisResult.best_model}</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium">Best AUC Score</p>
                      <p className="text-2xl font-bold">{(analysisResult.best_auc_score * 100).toFixed(1)}%</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium">Cross-Validation AUC</p>
                      <p className="text-lg">{(analysisResult.cv_auc_mean * 100).toFixed(1)}% ± {(analysisResult.cv_auc_std * 100).toFixed(2)}%</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium">Model Stability</p>
                      <p className="text-lg">{analysisResult.model_stability}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Model Rankings */}
              <Card>
                <CardHeader>
                  <CardTitle>Model Rankings</CardTitle>
                  <CardDescription>Performance comparison of all models</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {analysisResult.model_rankings.map(([model, score], index) => (
                      <div key={index} className="flex items-center justify-between">
                        <span className="font-medium">{model}</span>
                        <span className="text-lg font-semibold">{(score * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Top Features */}
              <Card>
                <CardHeader>
                  <CardTitle>Top Features</CardTitle>
                  <CardDescription>Most important features for prediction</CardDescription>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2">
                    {analysisResult.top_features.map((feature, index) => (
                      <li key={index} className="flex items-start">
                        <span className="mr-2">•</span>
                        <span>{feature}</span>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>

              {/* Recommendation */}
              <Card>
                <CardHeader>
                  <CardTitle>Recommendation</CardTitle>
                  <CardDescription>Model deployment suggestion</CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-lg">{analysisResult.recommendation}</p>
                </CardContent>
              </Card>

              {/* Debug Section */}
              <Card>
                <CardHeader>
                  <CardTitle>Debug Information</CardTitle>
                  <CardDescription>Raw data structure verification</CardDescription>
                </CardHeader>
                <CardContent>
                  <pre className="bg-gray-50 p-4 rounded-lg overflow-auto text-sm">
                    {JSON.stringify({
                      status: "success",
                      result: {
                        task_completed: analysisResult.task_completed,
                        best_model: analysisResult.best_model,
                        best_auc_score: analysisResult.best_auc_score,
                        cv_auc_mean: analysisResult.cv_auc_mean,
                        cv_auc_std: analysisResult.cv_auc_std,
                        model_rankings: analysisResult.model_rankings,
                        top_features: analysisResult.top_features,
                        model_stability: analysisResult.model_stability,
                        recommendation: analysisResult.recommendation
                      }
                    }, null, 2)}
                  </pre>
                </CardContent>
              </Card>
            </div>
          )}
        </div>
      </div>
    </div>
  )
} 