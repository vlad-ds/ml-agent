"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { useEffect, useState, use } from "react"

interface AnalysisPageProps {
  params: Promise<{
    directory: string
  }>
}

interface DatasetAnalysis {
  samples: number
  features: number
  feature_types: string
  target_balance: string
  data_quality: string
}

interface ModelPerformance {
  best_model: string
  test_auc: number
  test_accuracy: number
  cv_auc: string
  features_used: number
  features_removed: number
}

interface AnalysisResult {
  dataset_analysis: DatasetAnalysis
  model_performance: ModelPerformance
  key_insights: string[]
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

        const data = await response.json()
        
        // Validate the response structure
        if (!data.result || !data.result.dataset_analysis || !data.result.model_performance || !data.result.key_insights) {
          throw new Error('Invalid response structure from API')
        }

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

          {analysisResult && analysisResult.dataset_analysis && (
            <div className="space-y-6">
              {/* Dataset Analysis */}
              <Card>
                <CardHeader>
                  <CardTitle>Dataset Overview</CardTitle>
                  <CardDescription>Basic information about your dataset</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm font-medium">Samples</p>
                      <p className="text-2xl font-bold">{analysisResult.dataset_analysis.samples.toLocaleString()}</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium">Features</p>
                      <p className="text-2xl font-bold">{analysisResult.dataset_analysis.features}</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium">Feature Types</p>
                      <p className="text-lg">{analysisResult.dataset_analysis.feature_types}</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium">Target Balance</p>
                      <p className="text-lg">{analysisResult.dataset_analysis.target_balance}</p>
                    </div>
                    <div className="col-span-2">
                      <p className="text-sm font-medium">Data Quality</p>
                      <p className="text-lg">{analysisResult.dataset_analysis.data_quality}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Model Performance */}
              {analysisResult.model_performance && (
                <Card>
                  <CardHeader>
                    <CardTitle>Model Performance</CardTitle>
                    <CardDescription>Results from the best performing model</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-sm font-medium">Best Model</p>
                        <p className="text-lg font-semibold">{analysisResult.model_performance.best_model}</p>
                      </div>
                      <div>
                        <p className="text-sm font-medium">Test AUC</p>
                        <p className="text-2xl font-bold">{(analysisResult.model_performance.test_auc * 100).toFixed(1)}%</p>
                      </div>
                      <div>
                        <p className="text-sm font-medium">Test Accuracy</p>
                        <p className="text-2xl font-bold">{(analysisResult.model_performance.test_accuracy * 100).toFixed(1)}%</p>
                      </div>
                      <div>
                        <p className="text-sm font-medium">Cross-Validation AUC</p>
                        <p className="text-lg">{analysisResult.model_performance.cv_auc}</p>
                      </div>
                      <div>
                        <p className="text-sm font-medium">Features Used</p>
                        <p className="text-lg">{analysisResult.model_performance.features_used}</p>
                      </div>
                      <div>
                        <p className="text-sm font-medium">Features Removed</p>
                        <p className="text-lg">{analysisResult.model_performance.features_removed}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Key Insights */}
              {analysisResult.key_insights && (
                <Card>
                  <CardHeader>
                    <CardTitle>Key Insights</CardTitle>
                    <CardDescription>Important findings from the analysis</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-2">
                      {analysisResult.key_insights.map((insight, index) => (
                        <li key={index} className="flex items-start">
                          <span className="mr-2">•</span>
                          <span>{insight}</span>
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
} 